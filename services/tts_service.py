"""
Text-to-Speech service using VITS / MMS TTS models.

Generates speech audio from text in sentence-level chunks for streaming.
Runs inference in a thread pool to avoid blocking the event loop.
Supports immediate cancellation for barge-in interrupts.
"""

from __future__ import annotations

import asyncio
import io
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Optional

import numpy as np
import torch

from core.config import TTSSettings
from core.exceptions import TTSError
from core.logger import get_logger
from core.metrics import MetricsCollector

logger = get_logger(__name__)


class TTSService:
    """
    Enterprise TTS service with sentence-level streaming synthesis.

    Loads a VITS/MMS model at startup on GPU/CPU. Synthesizes text
    chunks from an asyncio queue and yields raw PCM audio bytes.
    """

    def __init__(self, settings: TTSSettings) -> None:
        self._settings = settings
        self._model: Optional[torch.nn.Module] = None
        self._tokenizer: Optional[object] = None
        self._device: Optional[torch.device] = None
        self._model_sample_rate: int = 16000  # Will be updated from model config
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts")
        self._loaded = False

    async def load_model(self) -> None:
        """Load TTS model and tokenizer onto the target device."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._load_model_sync)

    def _resolve_device(self) -> torch.device:
        device_str = self._settings.device
        if device_str == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("tts_device_auto_resolved", device="cuda")
            else:
                device = torch.device("cpu")
                logger.info("tts_device_auto_resolved", device="cpu")
        else:
            device = torch.device(device_str)
        return device

    def _load_model_sync(self) -> None:
        """Synchronous model loading (runs in thread pool)."""
        from transformers import AutoTokenizer, VitsModel

        model_name = self._settings.model_name
        self._device = self._resolve_device()

        logger.info("tts_model_loading", model=model_name, device=str(self._device))

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = VitsModel.from_pretrained(model_name)
            self._model = self._model.to(self._device)  # type: ignore[union-attr]
            self._model.eval()  # type: ignore[union-attr]

            # Extract native sample rate from model config
            if hasattr(self._model, 'config') and hasattr(self._model.config, 'sampling_rate'):
                self._model_sample_rate = self._model.config.sampling_rate

        except Exception as exc:
            raise TTSError(
                f"Failed to load TTS model '{model_name}'",
                model_name=model_name,
                original_error=str(exc),
            ) from exc

        self._loaded = True
        logger.info(
            "tts_model_loaded",
            model=model_name,
            device=str(self._device),
            native_sample_rate=self._model_sample_rate,
            output_sample_rate=self._settings.output_sample_rate,
        )

    async def unload_model(self) -> None:
        """Release model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
        self._executor.shutdown(wait=False)
        logger.info("tts_model_unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def synthesize_text(self, text: str) -> bytes:
        """
        Synthesize a single text string to PCM audio bytes.

        Args:
            text: Input text (Urdu) to synthesize.

        Returns:
            Raw 16-bit PCM audio bytes at the configured output sample rate.
        """
        if not self._loaded:
            raise TTSError("TTS model not loaded")

        if not text.strip():
            return b""

        if len(text) > self._settings.max_text_length:
            text = text[:self._settings.max_text_length]
            logger.warning("tts_text_truncated", max_length=self._settings.max_text_length)

        loop = asyncio.get_event_loop()
        start = time.monotonic()

        try:
            pcm_bytes = await asyncio.wait_for(
                loop.run_in_executor(self._executor, self._synthesize_sync, text),
                timeout=self._settings.inference_timeout,
            )
            latency = time.monotonic() - start
            MetricsCollector.record_tts_latency(latency)
            MetricsCollector.increment_call("tts_chunk")
            return pcm_bytes

        except asyncio.TimeoutError:
            MetricsCollector.increment_timeout("tts")
            MetricsCollector.increment_tts_error("timeout")
            raise TTSError("TTS inference timeout", timeout=self._settings.inference_timeout)
        except TTSError:
            raise
        except Exception as exc:
            MetricsCollector.increment_tts_error("inference_error")
            raise TTSError("TTS inference failed", original_error=str(exc)) from exc

    def _synthesize_sync(self, text: str) -> bytes:
        """Synchronous synthesis (runs in thread pool)."""
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._device is not None

        inputs = self._tokenizer(text, return_tensors="pt")  # type: ignore[operator]
        input_ids = inputs["input_ids"].to(self._device)

        with torch.no_grad():
            output = self._model(input_ids)  # type: ignore[operator]
            waveform = output.waveform[0].cpu().numpy()

        # Resample if model sample rate differs from output requirement
        if self._model_sample_rate != self._settings.output_sample_rate:
            waveform = self._resample_numpy(
                waveform,
                self._model_sample_rate,
                self._settings.output_sample_rate,
            )

        # Convert float32 [-1, 1] to int16 PCM
        waveform = np.clip(waveform, -1.0, 1.0)
        pcm_int16 = (waveform * 32767).astype(np.int16)
        return pcm_int16.tobytes()

    @staticmethod
    def _resample_numpy(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling using linear interpolation."""
        if orig_sr == target_sr:
            return audio
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def split_into_sentences(self, text: str) -> list[str]:
        """
        Split text at sentence boundaries for incremental synthesis.

        Uses configured delimiters to split Urdu text into synthesizable chunks.
        """
        delimiters = self._settings.sentence_delimiters
        sentences: list[str] = []
        current: list[str] = []

        for char in text:
            current.append(char)
            if char in delimiters:
                sentence = ''.join(current).strip()
                if sentence:
                    sentences.append(sentence)
                current = []

        # Remaining text without delimiter
        remainder = ''.join(current).strip()
        if remainder:
            sentences.append(remainder)

        return sentences

    async def synthesize_stream(
        self,
        text_queue: asyncio.Queue[str | None],
        cancel_event: asyncio.Event,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream audio synthesis from text chunks arriving on a queue.

        Reads text fragments from the queue, synthesizes each to audio,
        and yields PCM bytes. Supports immediate cancellation.

        Args:
            text_queue: Queue of text strings to synthesize. None signals end.
            cancel_event: Event to signal immediate cancellation.

        Yields:
            Raw 16-bit PCM audio bytes at the configured output sample rate.
        """
        accumulated_text = ""

        while not cancel_event.is_set():
            try:
                text_fragment = await asyncio.wait_for(text_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

            if text_fragment is None:
                # End of stream - synthesize any remaining text
                if accumulated_text.strip():
                    try:
                        audio = await self.synthesize_text(accumulated_text)
                        if audio:
                            yield audio
                    except TTSError:
                        logger.exception("tts_final_chunk_error")
                break

            if cancel_event.is_set():
                break

            accumulated_text += text_fragment

            # Check if we have a complete sentence to synthesize
            sentences = self.split_into_sentences(accumulated_text)
            if len(sentences) > 1:
                # Synthesize all complete sentences, keep the last (incomplete) one
                for sentence in sentences[:-1]:
                    if cancel_event.is_set():
                        return
                    try:
                        audio = await self.synthesize_text(sentence)
                        if audio:
                            yield audio
                    except TTSError:
                        logger.exception("tts_sentence_error", sentence_len=len(sentence))
                        continue

                accumulated_text = sentences[-1]

    @staticmethod
    def pcm_to_ulaw(pcm_bytes: bytes) -> bytes:
        """
        Convert 16-bit PCM audio to G.711 mu-law encoding.

        Args:
            pcm_bytes: Raw 16-bit signed PCM audio bytes.

        Returns:
            Mu-law encoded audio bytes (8-bit per sample).
        """
        BIAS = 0x84
        CLIP = 32635
        MU_MAX = 0x1FFF

        samples = struct.unpack(f'<{len(pcm_bytes) // 2}h', pcm_bytes)
        ulaw_bytes = bytearray(len(samples))

        exp_lut = [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]

        for i, sample in enumerate(samples):
            sign = (sample >> 8) & 0x80
            if sign != 0:
                sample = -sample
            if sample > CLIP:
                sample = CLIP
            sample = sample + BIAS
            exponent = exp_lut[(sample >> 7) & 0xFF]
            mantissa = (sample >> (exponent + 3)) & 0x0F
            ulaw_byte = ~(sign | (exponent << 4) | mantissa) & 0xFF
            ulaw_bytes[i] = ulaw_byte

        return bytes(ulaw_bytes)

    @staticmethod
    def ulaw_to_pcm(ulaw_bytes: bytes) -> bytes:
        """
        Convert G.711 mu-law audio to 16-bit PCM.

        Args:
            ulaw_bytes: Mu-law encoded audio bytes (8-bit per sample).

        Returns:
            Raw 16-bit signed PCM audio bytes.
        """
        exp_lut = [0, 132, 396, 924, 1980, 4092, 8316, 16764]
        pcm_samples = []

        for byte in ulaw_bytes:
            byte = ~byte & 0xFF
            sign = byte & 0x80
            exponent = (byte >> 4) & 0x07
            mantissa = byte & 0x0F
            sample = exp_lut[exponent] + (mantissa << (exponent + 3))
            if sign != 0:
                sample = -sample
            pcm_samples.append(sample)

        return struct.pack(f'<{len(pcm_samples)}h', *pcm_samples)
