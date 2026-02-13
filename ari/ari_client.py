"""
Asterisk ARI WebSocket client.

Connects to the Asterisk ARI event WebSocket with auto-reconnect
and exponential backoff. Supports REST-over-WebSocket for commands.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncGenerator, Optional

import websockets
import websockets.client
from websockets.exceptions import ConnectionClosed, WebSocketException

from core.config import ARISettings
from core.exceptions import ARICommandError, ARIConnectionError
from core.logger import get_logger
from core.metrics import ERRORS_TOTAL

logger = get_logger(__name__)


class ARIClient:
    """
     ARI WebSocket client with auto-reconnect.

    Maintains a persistent WebSocket connection to Asterisk ARI.
    Supports receiving events and sending REST commands over the
    same WebSocket connection (ARI REST-over-WebSocket protocol).
    """

    def __init__(self, settings: ARISettings) -> None:
        self._settings = settings
        self._ws: Optional[websockets.client.WebSocketClientProtocol] = None
        self._connected = False
        self._reconnecting = False
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)
        self._pending_commands: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._receive_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        logger.info(
            "ari_client_initialized",
            ws_url=settings.ws_url,
            app_name=settings.app_name,
        )

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    async def connect(self) -> None:
        """
        Establish WebSocket connection to ARI.

        Raises:
            ARIConnectionError: If initial connection fails after retries.
        """
        await self._connect_ws()

    async def _connect_ws(self) -> None:
        """Internal WebSocket connection with authentication."""
        import base64

        settings = self._settings
        auth_str = base64.b64encode(
            f"{settings.username}:{settings.password}".encode()
        ).decode()

        headers = {
            "Authorization": f"Basic {auth_str}",
        }

        attempt = 0
        delay = settings.reconnect_interval

        while not self._shutdown_event.is_set():
            try:
                self._ws = await websockets.connect(
                    settings.ws_url,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=10 * 1024 * 1024,  # 10MB max message
                )
                self._connected = True
                self._reconnecting = False

                logger.info("ari_connected", url=settings.ws_url)

                # Start receive loop
                if self._receive_task is not None:
                    self._receive_task.cancel()
                self._receive_task = asyncio.create_task(
                    self._receive_loop(), name="ari_receive"
                )
                return

            except (WebSocketException, OSError, ConnectionRefusedError) as exc:
                attempt += 1
                ERRORS_TOTAL.labels(category="ari").inc()

                if (
                    settings.reconnect_max_attempts > 0
                    and attempt >= settings.reconnect_max_attempts
                ):
                    raise ARIConnectionError(
                        f"ARI connection failed after {attempt} attempts",
                        url=settings.ws_url,
                        attempts=attempt,
                    ) from exc

                logger.warning(
                    "ari_connect_retry",
                    attempt=attempt,
                    delay=delay,
                    error=str(exc),
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, settings.reconnect_max_interval)

    async def _receive_loop(self) -> None:
        """
        Continuously receive messages from ARI WebSocket.

        Routes events to the event queue and command responses
        to their pending futures.
        """
        assert self._ws is not None

        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning("ari_invalid_json", raw=message[:200])
                    continue

                # Check if this is a REST-over-WebSocket response
                request_id = data.get("request_id")
                if request_id and request_id in self._pending_commands:
                    future = self._pending_commands.pop(request_id)
                    if not future.done():
                        future.set_result(data)
                    continue

                # Regular ARI event
                try:
                    self._event_queue.put_nowait(data)
                except asyncio.QueueFull:
                    logger.warning("ari_event_queue_full")

        except ConnectionClosed as exc:
            logger.warning("ari_connection_closed", code=exc.code, reason=str(exc.reason))
        except Exception:
            logger.exception("ari_receive_error")
        finally:
            self._connected = False
            if not self._shutdown_event.is_set():
                asyncio.create_task(self._auto_reconnect())

    async def _auto_reconnect(self) -> None:
        """Auto-reconnect after WebSocket disconnection."""
        if self._reconnecting or self._shutdown_event.is_set():
            return

        self._reconnecting = True
        logger.info("ari_auto_reconnect_starting")

        # Fail all pending commands
        for req_id, future in self._pending_commands.items():
            if not future.done():
                future.set_exception(
                    ARIConnectionError("ARI connection lost during command")
                )
        self._pending_commands.clear()

        try:
            await self._connect_ws()
        except ARIConnectionError:
            logger.error("ari_auto_reconnect_failed")

    async def disconnect(self) -> None:
        """Gracefully disconnect from ARI."""
        self._shutdown_event.set()

        if self._receive_task is not None:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass

        self._connected = False
        logger.info("ari_disconnected")

    async def get_events(self) -> AsyncGenerator[dict[str, Any], None]:
        """
        Async generator that yields ARI events from the event queue.

        Yields:
            Parsed ARI event dictionaries.
        """
        while not self._shutdown_event.is_set():
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=1.0
                )
                yield event
            except asyncio.TimeoutError:
                continue

    async def send_command(
        self,
        method: str,
        uri: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Send an ARI REST command over the WebSocket.

        Uses the ARI REST-over-WebSocket protocol with request/response
        correlation via request_id.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            uri: ARI REST URI (e.g., /channels/{id}/answer)
            body: Optional request body.

        Returns:
            Parsed response dictionary.

        Raises:
            ARICommandError: On command failure or timeout.
            ARIConnectionError: If not connected.
        """
        if not self.is_connected or self._ws is None:
            raise ARIConnectionError("Not connected to ARI")

        request_id = str(uuid.uuid4())
        command = {
            "request_id": request_id,
            "type": "RESTRequest",
            "method": method,
            "uri": uri,
        }
        if body:
            command["body"] = body

        # Create future for response
        future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        self._pending_commands[request_id] = future

        try:
            await self._ws.send(json.dumps(command))

            result = await asyncio.wait_for(
                future,
                timeout=self._settings.command_timeout,
            )

            # Check for error response
            status = result.get("status_code", 200)
            if status >= 400:
                raise ARICommandError(
                    f"ARI command failed: {method} {uri}",
                    status_code=status,
                    response=result,
                )

            return result

        except asyncio.TimeoutError:
            self._pending_commands.pop(request_id, None)
            raise ARICommandError(
                f"ARI command timed out: {method} {uri}",
                timeout=self._settings.command_timeout,
            )
        except ARICommandError:
            raise
        except Exception as exc:
            self._pending_commands.pop(request_id, None)
            raise ARICommandError(
                f"ARI command error: {method} {uri}",
                original_error=str(exc),
            ) from exc

    # --- Convenience Methods ---

    async def answer_channel(self, channel_id: str) -> dict[str, Any]:
        """Answer a channel."""
        return await self.send_command("POST", f"/channels/{channel_id}/answer")

    async def hangup_channel(self, channel_id: str, reason: str = "normal") -> dict[str, Any]:
        """Hang up a channel."""
        return await self.send_command(
            "DELETE", f"/channels/{channel_id}", body={"reason": reason}
        )

    async def create_bridge(self, bridge_type: str = "mixing") -> dict[str, Any]:
        """Create a new bridge."""
        return await self.send_command(
            "POST", "/bridges", body={"type": bridge_type}
        )

    async def add_channel_to_bridge(
        self, bridge_id: str, channel_id: str
    ) -> dict[str, Any]:
        """Add a channel to a bridge."""
        return await self.send_command(
            "POST",
            f"/bridges/{bridge_id}/addChannel",
            body={"channel": channel_id},
        )

    async def create_external_media(
        self,
        external_host: str,
        fmt: str = "ulaw",
        encapsulation: str = "rtp",
        direction: str = "both",
    ) -> dict[str, Any]:
        """
        Create an ExternalMedia channel for RTP bridging.

        Args:
            external_host: Host:port for RTP media (e.g., "10.0.0.1:10000")
            fmt: Audio codec format.
            encapsulation: Transport encapsulation (rtp).
            direction: Media direction (in, out, both).

        Returns:
            ARI response with channel details.
        """
        return await self.send_command(
            "POST",
            "/channels/externalMedia",
            body={
                "app": self._settings.app_name,
                "external_host": external_host,
                "format": fmt,
                "encapsulation": encapsulation,
                "direction": direction,
            },
        )

    async def remove_channel_from_bridge(
        self, bridge_id: str, channel_id: str
    ) -> dict[str, Any]:
        """Remove a channel from a bridge."""
        return await self.send_command(
            "POST",
            f"/bridges/{bridge_id}/removeChannel",
            body={"channel": channel_id},
        )

    async def destroy_bridge(self, bridge_id: str) -> dict[str, Any]:
        """Destroy a bridge."""
        return await self.send_command("DELETE", f"/bridges/{bridge_id}")
