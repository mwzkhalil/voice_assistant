"""
ARI/RTP audio transport implementation.

Wraps the existing RTPBridge to conform to the AudioTransport interface.
All ARI-specific logic (ExternalMedia, bridges) remains in call_controller.
This adapter only handles the audio I/O contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from transport.base import AudioTransport

if TYPE_CHECKING:
    from ari.rtp_bridge import RTPBridge


class ARITransport(AudioTransport):
    """
    AudioTransport adapter for ARI/RTP-based telephony audio.

    Delegates all audio I/O to the underlying RTPBridge.
    No business logic resides here.
    """

    def __init__(self, rtp_bridge: RTPBridge) -> None:
        self._bridge = rtp_bridge

    async def start(self) -> None:
        """Start the underlying RTP bridge."""
        await self._bridge.start()

    async def stop(self) -> None:
        """Stop the underlying RTP bridge."""
        await self._bridge.stop()

    async def read_audio(self, frame_size: int) -> bytes | None:
        """Read decoded PCM audio from the RTP bridge."""
        return await self._bridge.read_audio(frame_size)

    async def write_audio(self, pcm_data: bytes) -> None:
        """Write PCM audio out through the RTP bridge."""
        await self._bridge.write_audio(pcm_data)

    def stop_output(self) -> None:
        """Immediately halt RTP output for barge-in."""
        self._bridge.stop_output()

    def resume_output(self) -> None:
        """Resume RTP output."""
        self._bridge.resume_output()

    @property
    def rtp_bridge(self) -> RTPBridge:
        """Access the underlying RTP bridge for ARI-specific operations."""
        return self._bridge
