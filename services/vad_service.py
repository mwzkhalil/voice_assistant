"""
Voice Activity Detection service.

Wraps WebRTC VAD with state tracking for speech boundary detection.
Determines when a user starts and stops speaking for ASR segmentation
and barge-in interrupt detection.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import webrtcvad

from core.config import VADSettings
from core.logger import get_logger

logger = get_logger(__name__)


class VADEvent(enum.Enum):
    """Events emitted by the VAD state tracker."""
    SILENCE = "silence"
    SPEECH = "speech"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass
class VADResult:
    """Result from processing a single VAD frame."""
    event: VADEvent
    is_speech: bool
    speech_duration_ms: int
    silence_duration_ms: int
    should_trigger_asr: bool  # True when enough speech accumulated
    should_finalize: bool  # True when silence threshold reached after speech


@dataclass
class VADState:
    """Internal mutable state for the VAD tracker."""
    is_speaking: bool = False
    speech_frames: int = 0
    silence_frames: int = 0
    total_speech_ms: int = 0
    total_silence_ms: int = 0
    triggered: bool = False


class VADService:
    """
    Voice Activity Detection with state-based speech boundary tracking.

    Uses WebRTC VAD for frame-level speech detection with configurable
    aggressiveness. Tracks speech/silence durations to determine
    utterance boundaries.

    Thread-safe for single-call usage. Each call should get its own
    VADService instance or call create_tracker() for isolated state.
    """

    def __init__(self, settings: VADSettings) -> None:
        self._settings = settings
        self._vad = webrtcvad.Vad(settings.aggressiveness)
        self._frame_duration_ms = settings.frame_duration_ms
        self._silence_threshold_ms = settings.silence_threshold_ms
        self._speech_threshold_ms = settings.speech_threshold_ms
        self._sample_rate = settings.sample_rate

        # Derived constants
        self._frame_size_bytes = 2 * (self._sample_rate * self._frame_duration_ms // 1000)

        logger.info(
            "vad_service_initialized",
            aggressiveness=settings.aggressiveness,
            frame_duration_ms=self._frame_duration_ms,
            silence_threshold_ms=self._silence_threshold_ms,
            speech_threshold_ms=self._speech_threshold_ms,
            frame_size_bytes=self._frame_size_bytes,
        )

    @property
    def frame_duration_ms(self) -> int:
        return self._frame_duration_ms

    @property
    def frame_size_bytes(self) -> int:
        return self._frame_size_bytes

    def create_tracker(self) -> VADTracker:
        """Create an isolated VAD state tracker for a single call."""
        return VADTracker(
            vad=self._vad,
            frame_duration_ms=self._frame_duration_ms,
            silence_threshold_ms=self._silence_threshold_ms,
            speech_threshold_ms=self._speech_threshold_ms,
            sample_rate=self._sample_rate,
        )


class VADTracker:
    """
    Per-call VAD state tracker.

    Maintains speech/silence state for a single call session.
    Each call must use its own tracker instance.
    """

    def __init__(
        self,
        vad: webrtcvad.Vad,
        frame_duration_ms: int,
        silence_threshold_ms: int,
        speech_threshold_ms: int,
        sample_rate: int,
    ) -> None:
        self._vad = vad
        self._frame_duration_ms = frame_duration_ms
        self._silence_threshold_ms = silence_threshold_ms
        self._speech_threshold_ms = speech_threshold_ms
        self._sample_rate = sample_rate
        self._state = VADState()

        # Threshold in frame counts
        self._silence_frames_threshold = silence_threshold_ms // frame_duration_ms
        self._speech_frames_threshold = speech_threshold_ms // frame_duration_ms

    @property
    def is_speaking(self) -> bool:
        return self._state.is_speaking

    @property
    def speech_duration_ms(self) -> int:
        return self._state.total_speech_ms

    @property
    def silence_duration_ms(self) -> int:
        return self._state.total_silence_ms

    def reset(self) -> None:
        """Reset VAD state for a new utterance cycle."""
        self._state = VADState()

    def process_frame(self, audio_frame: bytes) -> VADResult:
        """
        Process a single audio frame through VAD.

        Args:
            audio_frame: Raw PCM 16-bit audio frame at configured sample rate.
                        Must be exactly frame_duration_ms worth of audio.

        Returns:
            VADResult with the current state and any triggered events.
        """
        expected_size = 2 * (self._sample_rate * self._frame_duration_ms // 1000)
        if len(audio_frame) != expected_size:
            # Pad or truncate to expected size
            if len(audio_frame) < expected_size:
                audio_frame = audio_frame + b'\x00' * (expected_size - len(audio_frame))
            else:
                audio_frame = audio_frame[:expected_size]

        is_speech = self._vad.is_speech(audio_frame, self._sample_rate)

        state = self._state
        event: VADEvent
        should_trigger_asr = False
        should_finalize = False

        if is_speech:
            state.speech_frames += 1
            state.total_speech_ms += self._frame_duration_ms
            state.silence_frames = 0
            state.total_silence_ms = 0

            if not state.is_speaking and state.speech_frames >= self._speech_frames_threshold:
                # Transition: silence -> speaking
                state.is_speaking = True
                state.triggered = True
                event = VADEvent.SPEECH_START
                should_trigger_asr = True
            else:
                event = VADEvent.SPEECH
                if state.is_speaking:
                    should_trigger_asr = True
        else:
            state.silence_frames += 1
            state.total_silence_ms += self._frame_duration_ms
            state.speech_frames = 0

            if state.is_speaking and state.silence_frames >= self._silence_frames_threshold:
                # Transition: speaking -> silence (utterance end)
                state.is_speaking = False
                should_finalize = True
                event = VADEvent.SPEECH_END
                state.triggered = False
            else:
                event = VADEvent.SILENCE

        return VADResult(
            event=event,
            is_speech=is_speech,
            speech_duration_ms=state.total_speech_ms,
            silence_duration_ms=state.total_silence_ms,
            should_trigger_asr=should_trigger_asr and state.is_speaking,
            should_finalize=should_finalize,
        )
