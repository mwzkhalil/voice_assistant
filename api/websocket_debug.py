"""
WebSocket debug endpoint for real-time call monitoring.

WS /ws/debug - Streams real-time call events, state transitions, and metrics.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from core.dependency_injection import get_container
from core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["debug"])

# Connected debug clients
_debug_clients: set[WebSocket] = set()
_debug_lock = asyncio.Lock()

# Event buffer for broadcasting
_event_buffer: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)


async def broadcast_debug_event(event: dict[str, Any]) -> None:
    """
    Broadcast a debug event to all connected WebSocket clients.

    Called from pipeline, session manager, or call controller
    to push real-time updates.
    """
    event["timestamp"] = time.time()
    try:
        _event_buffer.put_nowait(event)
    except asyncio.QueueFull:
        pass


@router.websocket("/ws/debug")
async def websocket_debug(
    websocket: WebSocket,
    token: str = Query(default="", description="Auth token"),
) -> None:
    """
    WebSocket endpoint for real-time debug monitoring.

    Streams call events, state transitions, and system metrics.
    Rate-limited to prevent overwhelming slow clients.
    """
    # Simple token validation (production should use proper auth)
    expected_token = "debug"  # In production, read from env
    if token and token != expected_token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    logger.info("debug_client_connected", client=str(websocket.client))

    async with _debug_lock:
        _debug_clients.add(websocket)

    try:
        # Send initial state snapshot
        container = get_container()
        if container.is_initialized:
            await websocket.send_json({
                "type": "snapshot",
                "active_calls": container.call_controller.active_call_count,
                "ari_connected": container.ari_client.is_connected,
                "asr_loaded": container.asr_service.is_loaded,
                "tts_loaded": container.tts_service.is_loaded,
                "timestamp": time.time(),
            })

        # Stream events with rate limiting
        last_send = 0.0
        min_interval = 0.05  # 50ms minimum between sends

        while True:
            try:
                # Check for incoming messages (ping/pong, close)
                try:
                    msg = await asyncio.wait_for(
                        websocket.receive_text(), timeout=0.01
                    )
                    if msg == "ping":
                        await websocket.send_text("pong")
                except asyncio.TimeoutError:
                    pass

                # Send buffered events
                events: list[dict[str, Any]] = []
                while not _event_buffer.empty() and len(events) < 50:
                    try:
                        events.append(_event_buffer.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                if events:
                    now = time.time()
                    if now - last_send >= min_interval:
                        await websocket.send_json({
                            "type": "events",
                            "data": events,
                        })
                        last_send = now

                # Periodic stats push
                if container.is_initialized:
                    await asyncio.sleep(1.0)
                    await websocket.send_json({
                        "type": "stats",
                        "active_calls": container.call_controller.active_call_count,
                        "timestamp": time.time(),
                    })
                else:
                    await asyncio.sleep(1.0)

            except WebSocketDisconnect:
                break
            except Exception:
                logger.exception("debug_ws_error")
                break

    finally:
        async with _debug_lock:
            _debug_clients.discard(websocket)
        logger.info("debug_client_disconnected", client=str(websocket.client))
