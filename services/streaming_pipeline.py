"""
Per-call streaming pipeline orchestrator.

Manages the full Audio-in -> VAD -> ASR -> LLM -> TTS -> Audio-out pipeline.
Connected via bounded asyncio queues with backpressure.
Supports barge-in interrupts and graceful cancellation.

Transport-agnostic: works with any AudioTransport implementation
(ARI/RTP, WebSocket, etc.).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from core.exceptions import PipelineError, PipelineTimeout
from core.logger import get_logger, set_call_id
from core.metrics import MetricsCollector
from services.session_manager import CallState

if TYPE_CHECKING:
    from transport.base import AudioTransport
    from services.asr_service import ASRService
    from services.llm_service import LLMService
    from services.session_manager import SessionManager
    from services.tts_service import TTSService
    from services.vad_service import VADService, VADTracker

logger = get_logger(__name__)

# Sentinel to signal end-of-stream on queues
STREAM_END: None = None

# Queue sizes for backpressure
ASR_AUDIO_QUEUE_SIZE = 100    # ~3 seconds at 30ms frames
ASR_TEXT_QUEUE_SIZE = 10
LLM_TEXT_QUEUE_SIZE = 50
TTS_AUDIO_QUEUE_SIZE = 20


class StreamingPipeline:
    """
    Isolated per-call streaming pipeline (transport-agnostic).

    Orchestrates:
        Audio-in -> VAD -> ASR -> LLM -> TTS -> Audio-out

    Each stage runs as an independent asyncio task connected by bounded queues.
    Supports barge-in interrupts that cancel TTS/LLM and reset to LISTENING.
    Works with any AudioTransport implementation.
    """

    def __init__(
        self,
        call_id: str,
        transport: AudioTransport,
        asr_service: ASRService,
        llm_service: LLMService,
        tts_service: TTSService,
        vad_service: VADService,
        session_manager: SessionManager,
    ) -> None:
        self._call_id = call_id
        self._transport = transport
        self._asr = asr_service
        self._llm = llm_service
        self._tts = tts_service
        self._vad_service = vad_service
        self._session_mgr = session_manager

        # Per-call VAD tracker
        self._vad_tracker: Optional[VADTracker] = None

        # Inter-stage queues
        self._asr_audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(
            maxsize=ASR_AUDIO_QUEUE_SIZE
        )
        self._asr_text_queue: asyncio.Queue[str | None] = asyncio.Queue(
            maxsize=ASR_TEXT_QUEUE_SIZE
        )
        self._llm_text_queue: asyncio.Queue[str | None] = asyncio.Queue(
            maxsize=LLM_TEXT_QUEUE_SIZE
        )
        self._tts_audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(
            maxsize=TTS_AUDIO_QUEUE_SIZE
        )

        # Cancellation events
        self._cancel_event = asyncio.Event()
        self._interrupt_event = asyncio.Event()
        self._llm_cancel = asyncio.Event()
        self._tts_cancel = asyncio.Event()

        # Task references
        self._tasks: list[asyncio.Task] = []
        self._running = False

        # Latency tracking
        self._speech_end_time: float = 0.0

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start all pipeline stages as concurrent tasks."""
        if self._running:
            return

        set_call_id(self._call_id)
        self._vad_tracker = self._vad_service.create_tracker()
        self._running = True

        logger.info("pipeline_start", call_id=self._call_id)

        self._tasks = [
            asyncio.create_task(self._audio_ingest_loop(), name=f"audio_in_{self._call_id}"),
            asyncio.create_task(self._asr_loop(), name=f"asr_{self._call_id}"),
            asyncio.create_task(self._llm_loop(), name=f"llm_{self._call_id}"),
            asyncio.create_task(self._tts_loop(), name=f"tts_{self._call_id}"),
            asyncio.create_task(self._audio_output_loop(), name=f"audio_out_{self._call_id}"),
        ]

        # Monitor tasks for unexpected failures
        for task in self._tasks:
            task.add_done_callback(self._task_done_callback)

    def _task_done_callback(self, task: asyncio.Task) -> None:
        """Handle unexpected task completion."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "pipeline_task_failed",
                call_id=self._call_id,
                task_name=task.get_name(),
                error=str(exc),
            )

    async def stop(self) -> None:
        """Gracefully stop all pipeline stages."""
        if not self._running:
            return

        logger.info("pipeline_stop", call_id=self._call_id)
        self._running = False
        self._cancel_event.set()
        self._llm_cancel.set()
        self._tts_cancel.set()

        # Signal end-of-stream on all queues
        for q in (self._asr_audio_queue, self._asr_text_queue,
                   self._llm_text_queue, self._tts_audio_queue):
            try:
                q.put_nowait(STREAM_END)
            except asyncio.QueueFull:
                pass

        # Cancel all tasks with timeout
        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._tasks.clear()
        logger.info("pipeline_stopped", call_id=self._call_id)

    async def interrupt(self) -> None:
        """
        Handle barge-in interrupt.

        Cancels current LLM and TTS processing, flushes queues,
        and resets to LISTENING state.
        """
        logger.info("pipeline_interrupt", call_id=self._call_id)
        MetricsCollector.increment_pipeline_interrupt()

        # Signal cancellation to LLM and TTS
        self._llm_cancel.set()
        self._tts_cancel.set()
        self._interrupt_event.set()

        # Flush queues
        for q in (self._asr_text_queue, self._llm_text_queue, self._tts_audio_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break

        # Stop transport output immediately
        self._transport.stop_output()

        # Transition state
        try:
            await self._session_mgr.transition_state(
                self._call_id, CallState.INTERRUPTED
            )
            await self._session_mgr.transition_state(
                self._call_id, CallState.LISTENING
            )
        except Exception:
            logger.exception("pipeline_interrupt_state_error", call_id=self._call_id)

        # Reset cancellation events for next turn
        self._llm_cancel.clear()
        self._tts_cancel.clear()
        self._interrupt_event.clear()

        # Reset VAD for new utterance
        if self._vad_tracker:
            self._vad_tracker.reset()

        logger.info("pipeline_interrupt_complete", call_id=self._call_id)

    # ---- Pipeline Stages ----

    async def _audio_ingest_loop(self) -> None:
        """
        Stage 1: Read audio from transport, run VAD, feed ASR queue.

        Reads PCM audio from the transport (RTP or WebSocket), runs
        frame-level VAD, and routes audio frames to the ASR queue.
        """
        assert self._vad_tracker is not None
        frame_size = self._vad_service.frame_size_bytes

        while not self._cancel_event.is_set():
            try:
                pcm_frame = await asyncio.wait_for(
                    self._transport.read_audio(frame_size),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                continue
            except Exception:
                if self._cancel_event.is_set():
                    break
                logger.exception("audio_ingest_error", call_id=self._call_id)
                continue

            if pcm_frame is None or self._cancel_event.is_set():
                break

            # Run VAD
            vad_result = self._vad_tracker.process_frame(pcm_frame)

            # Check for barge-in during SPEAKING
            session = await self._session_mgr.get_session(self._call_id)
            if session.state == CallState.SPEAKING and vad_result.is_speech:
                # Barge-in detected
                await self.interrupt()
                continue

            # Feed audio to ASR when speech detected
            if vad_result.should_trigger_asr:
                try:
                    self._asr_audio_queue.put_nowait(pcm_frame)
                except asyncio.QueueFull:
                    logger.warning("asr_audio_queue_full", call_id=self._call_id)

            # End of utterance - finalize ASR
            if vad_result.should_finalize:
                self._speech_end_time = time.monotonic()
                try:
                    self._asr_audio_queue.put_nowait(STREAM_END)
                except asyncio.QueueFull:
                    pass

                # Transition to THINKING
                try:
                    await self._session_mgr.transition_state(
                        self._call_id, CallState.THINKING
                    )
                except Exception:
                    logger.exception(
                        "pipeline_state_transition_error",
                        call_id=self._call_id,
                        target="THINKING",
                    )

    async def _asr_loop(self) -> None:
        """
        Stage 2: Run ASR on audio chunks and produce transcripts.
        """
        while not self._cancel_event.is_set():
            transcript_parts: list[str] = []

            async for text in self._asr.transcribe_stream(
                self._asr_audio_queue, self._cancel_event
            ):
                if self._cancel_event.is_set():
                    break
                transcript_parts.append(text)
                logger.debug(
                    "asr_partial_transcript",
                    call_id=self._call_id,
                    text=text,
                )

            if self._cancel_event.is_set():
                break

            # Combine partial transcripts into final
            full_transcript = " ".join(transcript_parts).strip()
            if full_transcript:
                logger.info(
                    "asr_final_transcript",
                    call_id=self._call_id,
                    text=full_transcript,
                )

                # Store user message in session
                try:
                    session = await self._session_mgr.get_session(self._call_id)
                    session.add_message("user", full_transcript)
                    await self._session_mgr.update_session(session)
                except Exception:
                    logger.exception("asr_session_update_error", call_id=self._call_id)

                # Forward to LLM
                try:
                    self._asr_text_queue.put_nowait(full_transcript)
                except asyncio.QueueFull:
                    logger.warning("asr_text_queue_full", call_id=self._call_id)

            # Re-create ASR audio queue for next utterance
            self._asr_audio_queue = asyncio.Queue(maxsize=ASR_AUDIO_QUEUE_SIZE)

    async def _llm_loop(self) -> None:
        """
        Stage 3: Stream LLM response tokens to TTS queue.
        """
        while not self._cancel_event.is_set():
            try:
                transcript = await asyncio.wait_for(
                    self._asr_text_queue.get(), timeout=2.0
                )
            except asyncio.TimeoutError:
                continue

            if transcript is None or self._cancel_event.is_set():
                break

            # Get session history
            try:
                session = await self._session_mgr.get_session(self._call_id)
                history = session.get_history_for_llm()
            except Exception:
                history = []

            # Stream LLM tokens to TTS
            self._llm_cancel.clear()
            response_text = ""

            try:
                async for token in self._llm.generate_stream(
                    transcript, history, self._llm_cancel
                ):
                    if self._cancel_event.is_set() or self._llm_cancel.is_set():
                        break

                    response_text += token
                    try:
                        self._llm_text_queue.put_nowait(token)
                    except asyncio.QueueFull:
                        logger.warning("llm_text_queue_full", call_id=self._call_id)
                        await self._llm_text_queue.put(token)

            except Exception:
                logger.exception("llm_generation_error", call_id=self._call_id)
                continue

            # Signal end of LLM output
            try:
                self._llm_text_queue.put_nowait(STREAM_END)
            except asyncio.QueueFull:
                await self._llm_text_queue.put(STREAM_END)

            # Store assistant response in session
            if response_text:
                try:
                    session = await self._session_mgr.get_session(self._call_id)
                    session.add_message("assistant", response_text)
                    await self._session_mgr.update_session(session)
                except Exception:
                    logger.exception("llm_session_update_error", call_id=self._call_id)

    async def _tts_loop(self) -> None:
        """
        Stage 4: Synthesize TTS audio from LLM text tokens.
        """
        while not self._cancel_event.is_set():
            self._tts_cancel.clear()
            first_audio = True

            async for audio_chunk in self._tts.synthesize_stream(
                self._llm_text_queue, self._tts_cancel
            ):
                if self._cancel_event.is_set() or self._tts_cancel.is_set():
                    break

                if first_audio:
                    first_audio = False
                    # Transition to SPEAKING
                    try:
                        await self._session_mgr.transition_state(
                            self._call_id, CallState.SPEAKING
                        )
                    except Exception:
                        logger.exception(
                            "pipeline_state_transition_error",
                            call_id=self._call_id,
                            target="SPEAKING",
                        )

                    # Record end-to-end latency
                    if self._speech_end_time > 0:
                        e2e = time.monotonic() - self._speech_end_time
                        MetricsCollector.record_pipeline_e2e(e2e)
                        logger.info(
                            "pipeline_e2e_latency",
                            call_id=self._call_id,
                            latency_seconds=e2e,
                        )

                # Send audio to RTP output queue
                try:
                    self._tts_audio_queue.put_nowait(audio_chunk)
                except asyncio.QueueFull:
                    logger.warning("tts_audio_queue_full", call_id=self._call_id)
                    await self._tts_audio_queue.put(audio_chunk)

            if self._cancel_event.is_set():
                break

            # Signal end of TTS output
            try:
                self._tts_audio_queue.put_nowait(STREAM_END)
            except asyncio.QueueFull:
                await self._tts_audio_queue.put(STREAM_END)

            # Re-create LLM text queue for next turn
            self._llm_text_queue = asyncio.Queue(maxsize=LLM_TEXT_QUEUE_SIZE)

    async def _audio_output_loop(self) -> None:
        """
        Stage 5: Send TTS audio out via transport (RTP or WebSocket).
        """
        while not self._cancel_event.is_set():
            try:
                audio_chunk = await asyncio.wait_for(
                    self._tts_audio_queue.get(), timeout=2.0
                )
            except asyncio.TimeoutError:
                continue

            if audio_chunk is None:
                # TTS output complete for this turn
                # Transition back to LISTENING
                try:
                    session = await self._session_mgr.get_session(self._call_id)
                    if session.state == CallState.SPEAKING:
                        await self._session_mgr.transition_state(
                            self._call_id, CallState.LISTENING
                        )
                except Exception:
                    logger.exception(
                        "pipeline_state_transition_error",
                        call_id=self._call_id,
                        target="LISTENING",
                    )

                # Reset VAD for new utterance
                if self._vad_tracker:
                    self._vad_tracker.reset()

                # Re-create TTS audio queue for next turn
                self._tts_audio_queue = asyncio.Queue(maxsize=TTS_AUDIO_QUEUE_SIZE)
                continue

            if self._cancel_event.is_set():
                break

            # Write audio to transport
            try:
                await self._transport.write_audio(audio_chunk)
            except Exception:
                logger.exception("audio_output_error", call_id=self._call_id)
