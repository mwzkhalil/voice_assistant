"""
Public realtime WebSocket endpoint.

WS /ws/realtime - Full duplex audio streaming with ASR/LLM/TTS pipeline.

Protocol:
    Client sends:  {"type": "audio_chunk", "data": "<base64_pcm_16bit_16khz>"}
    Server sends:  {"type": "audio_chunk", "data": "<base64_pcm_16bit_16khz>"}
                   {"type": "partial_transcript", "text": "..."}
                   {"type": "final_transcript", "text": "..."}
                   {"type": "state_change", "state": "LISTENING"}
                   {"type": "error", "message": "..."}
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from api.auth import (
    authenticate_websocket,
    register_session,
    unregister_session,
)
from core.config import get_settings
from core.dependency_injection import get_container
from core.exceptions import ResourceExhausted
from core.logger import get_logger, set_call_id, set_correlation_id
from core.metrics import MetricsCollector
from services.session_manager import CallState
from services.streaming_pipeline import StreamingPipeline
from transport.websocket_transport import WebSocketTransport

logger = get_logger(__name__)

router = APIRouter(tags=["realtime"])

# Active WS sessions: call_id -> (pipeline, transport, client_ip)
_active_sessions: dict[str, tuple[StreamingPipeline, WebSocketTransport, str]] = {}
_sessions_lock = asyncio.Lock()


@router.websocket("/ws/realtime")
async def websocket_realtime(
    websocket: WebSocket,
    api_key: str = Query(default="", alias="api_key"),
) -> None:
    """
    Public realtime voice assistant WebSocket endpoint.

    Full duplex streaming: client sends audio, server responds with
    audio + transcripts. Barge-in interrupts supported.
    """
    settings = get_settings()
    container = get_container()

    if not container.is_initialized:
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="Service not ready")
        return

    # Authenticate and rate-limit before accepting
    try:
        client_ip = await authenticate_websocket(websocket)
    except Exception:
        return  # WebSocket already closed by auth layer

    # Check global session limit
    async with _sessions_lock:
        if len(_active_sessions) >= settings.server.max_concurrent_calls:
            await websocket.close(code=4029, reason="Server at capacity")
            MetricsCollector.increment_call("rejected")
            return

    # Accept the WebSocket
    await websocket.accept()

    call_id = str(uuid.uuid4())
    set_call_id(call_id)
    set_correlation_id(call_id)

    logger.info(
        "ws_realtime_connected",
        call_id=call_id,
        client_ip=client_ip,
    )

    await register_session(client_ip)
    MetricsCollector.increment_call("answered")
    MetricsCollector.increment_active_calls()

    pipeline: StreamingPipeline | None = None
    transport: WebSocketTransport | None = None

    try:
        # Create session
        session = await container.session_manager.create_session(call_id, f"ws:{client_ip}")
        await container.session_manager.transition_state(call_id, CallState.ANSWERED)

        # Create WebSocket transport
        transport = WebSocketTransport(
            websocket=websocket,
            call_id=call_id,
            max_chunk_size=settings.auth.max_chunk_size_bytes,
            idle_timeout=settings.auth.idle_session_timeout,
        )

        # Create transport-agnostic pipeline
        pipeline = StreamingPipeline(
            call_id=call_id,
            transport=transport,
            asr_service=container.asr_service,
            llm_service=container.llm_service,
            tts_service=container.tts_service,
            vad_service=container.vad_service,
            session_manager=container.session_manager,
        )

        # Register active session
        async with _sessions_lock:
            _active_sessions[call_id] = (pipeline, transport, client_ip)

        # Start transport and pipeline
        await transport.start()

        # Transition to LISTENING and notify client
        await container.session_manager.transition_state(call_id, CallState.LISTENING)
        await transport.send_state_change("LISTENING")

        # Start pipeline
        await pipeline.start()

        logger.info("ws_realtime_pipeline_started", call_id=call_id)

        # Wait until transport closes (client disconnect, idle timeout, or error)
        while not transport._cancel_event.is_set():
            await asyncio.sleep(1.0)

            # Check idle timeout
            if transport.is_idle_expired:
                logger.info("ws_realtime_idle_timeout", call_id=call_id)
                await transport.send_error("Session idle timeout")
                break

            # Check max audio duration
            session = await container.session_manager.get_session(call_id)
            session_age = time.time() - session.created_at
            if session_age > settings.auth.max_audio_duration_seconds:
                logger.info("ws_realtime_max_duration", call_id=call_id)
                await transport.send_error("Maximum session duration exceeded")
                break

    except WebSocketDisconnect:
        logger.info("ws_realtime_client_disconnect", call_id=call_id)
    except Exception:
        logger.exception("ws_realtime_error", call_id=call_id)
    finally:
        # Cleanup
        logger.info("ws_realtime_cleanup", call_id=call_id)

        if pipeline is not None:
            try:
                await asyncio.wait_for(pipeline.stop(), timeout=5.0)
            except Exception:
                logger.exception("ws_pipeline_stop_error", call_id=call_id)

        if transport is not None:
            try:
                await transport.stop()
            except Exception:
                pass

        # Transition to TERMINATED
        try:
            await container.session_manager.transition_state(call_id, CallState.TERMINATED)
        except Exception:
            pass

        # Calculate duration
        try:
            session = await container.session_manager.get_session(call_id)
            if session.answered_at:
                duration = time.time() - session.answered_at
                MetricsCollector.record_call_duration(duration)
        except Exception:
            pass

        # Remove from active sessions
        async with _sessions_lock:
            _active_sessions.pop(call_id, None)

        await unregister_session(client_ip)
        MetricsCollector.decrement_active_calls()

        logger.info("ws_realtime_ended", call_id=call_id)


async def get_active_ws_session_count() -> int:
    """Return count of active WebSocket sessions."""
    async with _sessions_lock:
        return len(_active_sessions)


async def shutdown_all_ws_sessions() -> None:
    """Gracefully shutdown all active WebSocket sessions."""
    async with _sessions_lock:
        session_ids = list(_active_sessions.keys())

    for call_id in session_ids:
        async with _sessions_lock:
            entry = _active_sessions.get(call_id)
        if entry:
            pipeline, transport, _ = entry
            try:
                await transport.send_error("Server shutting down")
                await pipeline.stop()
                await transport.stop()
            except Exception:
                pass

    async with _sessions_lock:
        _active_sessions.clear()
