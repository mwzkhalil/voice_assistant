"""
Automatic Speech Recognition service.

Supports Wav2Vec2 (CTC) and Whisper (seq2seq) architectures with
auto-detection. Processes audio chunks from an asyncio queue and
yields partial transcripts. GPU-accelerated with CPU fallback.

All inference runs in a thread pool executor to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Optional

import numpy as np
import torch
import torchaudio

from core.config import ASRSettings
from core.exceptions import ASRError
from core.logger import get_logger
from core.metrics import MetricsCollector

logger = get_logger(__name__)


class ASRService:
    """
    Enterprise ASR service with chunked windowed inference.

    Loads a HuggingFace ASR model (auto-detects Wav2Vec2 CTC or Whisper)
    at startup and runs inference in a thread pool to keep the event
    loop non-blocking.
    """

    def __init__(self, settings: ASRSettings) -> None:
        self._settings = settings
        self._model: Optional[torch.nn.Module] = None
        self._processor: Optional[object] = None
        self._device: Optional[torch.device] = None
        self._model_type: Optional[str] = None  # "ctc" or "seq2seq"
        self._resampler: Optional[torchaudio.transforms.Resample] = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="asr")
        self._loaded = False

    async def load_model(self) -> None:
        """Load ASR model and processor onto the target device."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._load_model_sync)

    def _resolve_device(self) -> torch.device:
        """Resolve the target device based on config and availability."""
        device_str = self._settings.device
        if device_str == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("asr_device_auto_resolved", device="cuda")
            else:
                device = torch.device("cpu")
                logger.info("asr_device_auto_resolved", device="cpu")
        else:
            device = torch.device(device_str)
        return device

    def _load_model_sync(self) -> None:
        """Synchronous model loading (runs in thread pool)."""
        from transformers import AutoModelForCTC, AutoProcessor, WhisperForConditionalGeneration, WhisperProcessor

        model_name = self._settings.model_name
        self._device = self._resolve_device()

        logger.info("asr_model_loading", model=model_name, device=str(self._device))

        # Auto-detect model architecture
        try:
            self._processor = AutoProcessor.from_pretrained(model_name)
            self._model = AutoModelForCTC.from_pretrained(model_name)
            self._model_type = "ctc"
            logger.info("asr_model_type_detected", model_type="ctc")
        except (ValueError, OSError):
            try:
                self._processor = WhisperProcessor.from_pretrained(model_name)
                self._model = WhisperForConditionalGeneration.from_pretrained(model_name)
                self._model_type = "seq2seq"
                logger.info("asr_model_type_detected", model_type="seq2seq")
            except Exception as exc:
                raise ASRError(
                    f"Failed to load ASR model '{model_name}'",
                    model_name=model_name,
                    original_error=str(exc),
                ) from exc

        self._model = self._model.to(self._device)  # type: ignore[union-attr]
        self._model.eval()  # type: ignore[union-attr]

        # Pre-build resampler for 8kHz -> 16kHz (common telephony input)
        if self._settings.sample_rate != 8000:
            self._resampler = torchaudio.transforms.Resample(
                orig_freq=8000,
                new_freq=self._settings.sample_rate,
            )

        self._loaded = True
        logger.info(
            "asr_model_loaded",
            model=model_name,
            model_type=self._model_type,
            device=str(self._device),
        )

    async def unload_model(self) -> None:
        """Release model from GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
        self._executor.shutdown(wait=False)
        logger.info("asr_model_unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def resample_from_telephony(self, pcm_8khz: np.ndarray) -> np.ndarray:
        """Resample 8kHz PCM audio to the model's expected sample rate."""
        if self._settings.sample_rate == 8000:
            return pcm_8khz
        if self._resampler is None:
            self._resampler = torchaudio.transforms.Resample(
                orig_freq=8000, new_freq=self._settings.sample_rate
            )
        tensor = torch.from_numpy(pcm_8khz).float()
        resampled = self._resampler(tensor)
        return resampled.numpy()

    async def transcribe_chunk(self, audio: np.ndarray) -> str:
        """
        Transcribe a single audio chunk.

        Args:
            audio: Numpy float32 array of audio at model sample rate.

        Returns:
            Transcribed text string.
        """
        if not self._loaded:
            raise ASRError("ASR model not loaded")

        loop = asyncio.get_event_loop()
        start = time.monotonic()

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(self._executor, self._transcribe_sync, audio),
                timeout=self._settings.inference_timeout,
            )
            latency = time.monotonic() - start
            MetricsCollector.record_asr_latency(latency)
            return result

        except asyncio.TimeoutError:
            MetricsCollector.increment_timeout("asr")
            MetricsCollector.increment_asr_error("timeout")
            raise ASRError(
                "ASR inference timeout",
                timeout=self._settings.inference_timeout,
            )
        except ASRError:
            raise
        except Exception as exc:
            MetricsCollector.increment_asr_error("inference_error")
            raise ASRError(
                "ASR inference failed",
                original_error=str(exc),
            ) from exc

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        """Synchronous transcription (runs in thread pool)."""
        assert self._model is not None
        assert self._processor is not None
        assert self._device is not None

        # Normalize audio
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0

        input_values = self._processor(
            audio,
            sampling_rate=self._settings.sample_rate,
            return_tensors="pt",
        )

        # Move inputs to device
        input_dict = {
            k: v.to(self._device) if isinstance(v, torch.Tensor) else v
            for k, v in input_values.items()
        }

        with torch.no_grad():
            if self._model_type == "ctc":
                logits = self._model(**input_dict).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self._processor.batch_decode(predicted_ids)[0]
            else:
                # Whisper / seq2seq
                generated_ids = self._model.generate(
                    input_dict.get("input_features", input_dict.get("input_values")),
                    max_new_tokens=448,
                )
                transcription = self._processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

        return transcription.strip()

    async def transcribe_stream(
        self,
        audio_queue: asyncio.Queue[bytes | None],
        cancel_event: asyncio.Event,
    ) -> AsyncGenerator[str, None]:
        """
        Continuously transcribe audio chunks from a queue.

        Reads raw PCM 16-bit bytes from the queue, accumulates into
        chunks of configured duration, runs inference, and yields
        transcription results.

        Args:
            audio_queue: Queue of raw PCM 16-bit audio bytes. None signals end.
            cancel_event: Event to signal cancellation.

        Yields:
            Transcribed text strings.
        """
        sample_rate = self._settings.sample_rate
        chunk_samples = int(sample_rate * self._settings.chunk_duration_ms / 1000)
        overlap_samples = int(sample_rate * self._settings.chunk_overlap_ms / 1000)
        max_buffer_samples = int(sample_rate * self._settings.max_audio_buffer_seconds)

        audio_buffer = np.array([], dtype=np.float32)

        while not cancel_event.is_set():
            try:
                raw_bytes = await asyncio.wait_for(audio_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

            if raw_bytes is None:
                # End of stream - transcribe remaining buffer
                if len(audio_buffer) > sample_rate * 0.3:  # at least 300ms
                    try:
                        text = await self.transcribe_chunk(audio_buffer)
                        if text:
                            yield text
                    except ASRError:
                        logger.exception("asr_final_chunk_error")
                break

            # Convert bytes to float32 numpy array
            pcm_16bit = np.frombuffer(raw_bytes, dtype=np.int16)
            audio_float = pcm_16bit.astype(np.float32) / 32768.0

            # Resample if needed (telephony 8kHz -> model rate)
            if sample_rate != 8000 and self._resampler is not None:
                audio_float = self.resample_from_telephony(audio_float)

            audio_buffer = np.concatenate([audio_buffer, audio_float])

            # Enforce max buffer size
            if len(audio_buffer) > max_buffer_samples:
                audio_buffer = audio_buffer[-max_buffer_samples:]

            # Process when we have enough audio
            if len(audio_buffer) >= chunk_samples:
                chunk = audio_buffer[:chunk_samples]
                # Keep overlap for context continuity
                audio_buffer = audio_buffer[chunk_samples - overlap_samples:]

                try:
                    text = await self.transcribe_chunk(chunk)
                    if text:
                        yield text
                except ASRError:
                    logger.exception("asr_chunk_error")
                    continue
