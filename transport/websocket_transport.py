"""
WebSocket audio transport implementation.

Receives base64-encoded PCM audio chunks from WebSocket clients
and sends base64-encoded PCM audio chunks back. Supports transcript
metadata messages on the same connection.

No business logic resides in this module.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any, Callable, Coroutine, Optional

from fastapi import WebSocket

from core.logger import get_logger
from core.metrics import RTP_PACKETS_DROPPED, RTP_PACKETS_RECEIVED, RTP_PACKETS_SENT
from transport.base import AudioTransport

logger = get_logger(__name__)


class WebSocketTransport(AudioTransport):
    """
    AudioTransport backed by a FastAPI WebSocket connection.

    Protocol:
        Client -> Server:
            {"type": "audio_chunk", "data": "<base64_pcm_16bit_16khz>"}

        Server -> Client:
            {"type": "audio_chunk", "data": "<base64_pcm_16bit_16khz>"}
            {"type": "partial_transcript", "text": "...", "is_final": false}
            {"type": "final_transcript", "text": "...", "is_final": true}
            {"type": "state_change", "state": "LISTENING"}
            {"type": "error", "message": "..."}
    """

    def __init__(
        self,
        websocket: WebSocket,
        call_id: str,
        max_chunk_size: int = 32768,
        idle_timeout: float = 300.0,
    ) -> None:
        self._ws = websocket
        self._call_id = call_id
        self._max_chunk_size = max_chunk_size
        self._idle_timeout = idle_timeout

        # Inbound PCM queue (client -> pipeline)
        self._inbound_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        # Outbound PCM queue (pipeline -> client)
        self._outbound_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=100)

        self._output_stopped = False
        self._cancel_event = asyncio.Event()
        self._receive_task: Optional[asyncio.Task] = None
        self._send_task: Optional[asyncio.Task] = None
        self._last_activity: float = time.monotonic()

        # Optional transcript callback
        self._transcript_callback: Optional[
            Callable[[str, bool], Coroutine[Any, Any, None]]
        ] = None

        logger.info("ws_transport_created", call_id=call_id)

    @property
    def last_activity(self) -> float:
        return self._last_activity

    @property
    def is_idle_expired(self) -> bool:
        return (time.monotonic() - self._last_activity) > self._idle_timeout

    async def start(self) -> None:
        """Start receive and send background loops."""
        self._receive_task = asyncio.create_task(
            self._receive_loop(), name=f"ws_recv_{self._call_id}"
        )
        self._send_task = asyncio.create_task(
            self._send_loop(), name=f"ws_send_{self._call_id}"
        )
        logger.info("ws_transport_started", call_id=self._call_id)

    async def stop(self) -> None:
        """Stop transport and close WebSocket."""
        self._cancel_event.set()

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._send_task:
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass

        try:
            await self._ws.close()
        except Exception:
            pass

        logger.info("ws_transport_stopped", call_id=self._call_id)

    async def read_audio(self, frame_size: int) -> bytes | None:
        """Read PCM audio from the inbound queue."""
        if self._cancel_event.is_set():
            return None

        try:
            pcm = await asyncio.wait_for(
                self._inbound_queue.get(), timeout=2.0
            )
            # Pad or trim to expected frame size
            if len(pcm) < frame_size:
                pcm = pcm + b'\x00' * (frame_size - len(pcm))
            elif len(pcm) > frame_size:
                remainder = pcm[frame_size:]
                pcm = pcm[:frame_size]
                try:
                    self._inbound_queue.put_nowait(remainder)
                except asyncio.QueueFull:
                    pass
            return pcm
        except asyncio.TimeoutError:
            return None

    async def write_audio(self, pcm_data: bytes) -> None:
        """Queue PCM audio for sending to the client."""
        if self._output_stopped or self._cancel_event.is_set():
            return
        try:
            self._outbound_queue.put_nowait(pcm_data)
        except asyncio.QueueFull:
            RTP_PACKETS_DROPPED.labels(reason="ws_outbound_full").inc()

    def stop_output(self) -> None:
        """Immediately stop outbound audio (barge-in)."""
        self._output_stopped = True
        # Flush outbound queue
        while not self._outbound_queue.empty():
            try:
                self._outbound_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def resume_output(self) -> None:
        """Resume outbound audio."""
        self._output_stopped = False

    def set_transcript_callback(
        self,
        callback: Callable[[str, bool], Coroutine[Any, Any, None]] | None,
    ) -> None:
        """Set callback for pushing transcripts to the WebSocket client."""
        self._transcript_callback = callback

    async def send_transcript(self, text: str, is_final: bool = False) -> None:
        """Send a transcript message to the client."""
        try:
            msg_type = "final_transcript" if is_final else "partial_transcript"
            await self._ws.send_json({
                "type": msg_type,
                "text": text,
                "is_final": is_final,
                "timestamp": time.time(),
            })
        except Exception:
            logger.debug("ws_transcript_send_error", call_id=self._call_id)

    async def send_state_change(self, state: str) -> None:
        """Send a state change notification to the client."""
        try:
            await self._ws.send_json({
                "type": "state_change",
                "state": state,
                "timestamp": time.time(),
            })
        except Exception:
            logger.debug("ws_state_send_error", call_id=self._call_id)

    async def send_error(self, message: str) -> None:
        """Send an error message to the client."""
        try:
            await self._ws.send_json({
                "type": "error",
                "message": message,
                "timestamp": time.time(),
            })
        except Exception:
            pass

    # ---- Background Loops ----

    async def _receive_loop(self) -> None:
        """Read messages from WebSocket and route audio to inbound queue."""
        while not self._cancel_event.is_set():
            try:
                raw = await asyncio.wait_for(
                    self._ws.receive_text(), timeout=self._idle_timeout
                )
                self._last_activity = time.monotonic()
            except asyncio.TimeoutError:
                logger.info("ws_idle_timeout", call_id=self._call_id)
                self._cancel_event.set()
                break
            except Exception:
                if not self._cancel_event.is_set():
                    logger.info("ws_receive_closed", call_id=self._call_id)
                self._cancel_event.set()
                break

            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                logger.warning("ws_invalid_json", call_id=self._call_id)
                continue

            msg_type = msg.get("type")

            if msg_type == "audio_chunk":
                b64_data = msg.get("data", "")
                if not b64_data:
                    continue

                # Enforce max chunk size
                if len(b64_data) > self._max_chunk_size:
                    await self.send_error("Chunk size exceeds maximum")
                    continue

                try:
                    pcm_bytes = base64.b64decode(b64_data)
                except Exception:
                    await self.send_error("Invalid base64 audio data")
                    continue

                RTP_PACKETS_RECEIVED.inc()

                try:
                    self._inbound_queue.put_nowait(pcm_bytes)
                except asyncio.QueueFull:
                    RTP_PACKETS_DROPPED.labels(reason="ws_inbound_full").inc()

            elif msg_type == "ping":
                try:
                    await self._ws.send_json({"type": "pong", "timestamp": time.time()})
                except Exception:
                    pass

            elif msg_type == "stop":
                logger.info("ws_client_stop", call_id=self._call_id)
                self._cancel_event.set()
                break

    async def _send_loop(self) -> None:
        """Send queued audio and metadata to WebSocket client."""
        while not self._cancel_event.is_set():
            try:
                pcm = await asyncio.wait_for(
                    self._outbound_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            if pcm is None or self._cancel_event.is_set():
                break

            if self._output_stopped:
                continue

            try:
                b64_audio = base64.b64encode(pcm).decode("ascii")
                await self._ws.send_json({
                    "type": "audio_chunk",
                    "data": b64_audio,
                    "timestamp": time.time(),
                })
                RTP_PACKETS_SENT.inc()
            except Exception:
                if not self._cancel_event.is_set():
                    logger.info("ws_send_closed", call_id=self._call_id)
                self._cancel_event.set()
                break
