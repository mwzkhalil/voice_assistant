"""
Abstract audio transport interface.

All transport implementations (ARI/RTP, WebSocket, etc.) must conform
to this interface. The StreamingPipeline is transport-agnostic and
only interacts with audio through this abstraction.

No business logic is allowed in transport implementations.
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Coroutine


class AudioTransport(abc.ABC):
    """
    Abstract base class for bidirectional audio transport.

    Every transport must:
    - Push PCM 16-bit 16kHz audio into the pipeline via read_audio()
    - Receive PCM 16-bit 16kHz audio from the pipeline via write_audio()
    - Support immediate output stop for barge-in
    - Support graceful start/stop lifecycle

    Implementations must NOT contain any ASR, LLM, TTS, or session logic.
    """

    @abc.abstractmethod
    async def start(self) -> None:
        """Start the transport (bind sockets, accept connections, etc.)."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop the transport and release all resources."""

    @abc.abstractmethod
    async def read_audio(self, frame_size: int) -> bytes | None:
        """
        Read a PCM audio frame from the transport.

        Args:
            frame_size: Expected frame size in bytes (16-bit PCM at 16kHz).

        Returns:
            PCM audio bytes of exactly frame_size, or None if transport closed.
        """

    @abc.abstractmethod
    async def write_audio(self, pcm_data: bytes) -> None:
        """
        Write PCM audio data out through the transport.

        Args:
            pcm_data: Raw 16-bit PCM audio bytes at 16kHz.
        """

    @abc.abstractmethod
    def stop_output(self) -> None:
        """Immediately stop audio output (for barge-in interrupt)."""

    @abc.abstractmethod
    def resume_output(self) -> None:
        """Resume audio output after barge-in reset."""

    def set_transcript_callback(
        self,
        callback: Callable[[str, bool], Coroutine[Any, Any, None]] | None,
    ) -> None:
        """
        Set a callback for sending transcript updates to the client.

        Only relevant for transports that support metadata channels
        (e.g., WebSocket). Ignored by transports that don't (e.g., RTP).

        Args:
            callback: Async callable(text, is_final) or None to clear.
        """
        # Default: no-op. Override in transports that support it.
        pass
