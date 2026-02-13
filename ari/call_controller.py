"""
Top-level call lifecycle controller.

Manages the full lifecycle of each call from StasisStart to ChannelDestroyed.
Coordinates ARI channel operations, RTP bridge setup, and pipeline orchestration.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Optional

from core.config import AppSettings
from core.exceptions import (
    ARICommandError,
    ResourceExhausted,
    SessionError,
    VoiceAssistantError,
)
from core.logger import get_logger, set_call_id
from core.metrics import MetricsCollector
from services.session_manager import CallState

from ari.ari_client import ARIClient
from ari.event_router import EventRouter
from ari.rtp_bridge import RTPBridge, get_port_allocator

from services.asr_service import ASRService
from services.llm_service import LLMService
from services.session_manager import SessionManager
from services.streaming_pipeline import StreamingPipeline
from services.tts_service import TTSService
from services.vad_service import VADService

logger = get_logger(__name__)


class ActiveCall:
    """Holds all resources for a single active call."""

    __slots__ = (
        "call_id", "channel_id", "pipeline", "rtp_bridge",
        "bridge_id", "external_media_channel_id", "start_time",
    )

    def __init__(
        self,
        call_id: str,
        channel_id: str,
        pipeline: StreamingPipeline,
        rtp_bridge: RTPBridge,
        bridge_id: str,
        external_media_channel_id: str,
    ) -> None:
        self.call_id = call_id
        self.channel_id = channel_id
        self.pipeline = pipeline
        self.rtp_bridge = rtp_bridge
        self.bridge_id = bridge_id
        self.external_media_channel_id = external_media_channel_id
        self.start_time = time.monotonic()


class CallController:
    """
     call lifecycle controller.

    Handles:
    - Incoming call setup (answer, ExternalMedia, bridge, pipeline start)
    - Call teardown (pipeline stop, bridge destroy, session cleanup)
    - Graceful shutdown with active call draining
    - Resource limit enforcement (max concurrent calls)
    """

    def __init__(
        self,
        ari_client: ARIClient,
        event_router: EventRouter,
        session_manager: SessionManager,
        asr_service: ASRService,
        tts_service: TTSService,
        llm_service: LLMService,
        vad_service: VADService,
        settings: AppSettings,
    ) -> None:
        self._ari = ari_client
        self._router = event_router
        self._session_mgr = session_manager
        self._asr = asr_service
        self._tts = tts_service
        self._llm = llm_service
        self._vad = vad_service
        self._settings = settings

        # Active call registry: channel_id -> ActiveCall
        self._active_calls: dict[str, ActiveCall] = {}
        self._lock = asyncio.Lock()
        self._shutting_down = False

        self._port_allocator = get_port_allocator(settings.rtp)

        logger.info(
            "call_controller_initialized",
            max_concurrent=settings.server.max_concurrent_calls,
        )

    def register_handlers(self) -> None:
        """Register ARI event handlers with the event router."""
        self._router.register("StasisStart", self._on_stasis_start)
        self._router.register("StasisEnd", self._on_stasis_end)
        self._router.register("ChannelDestroyed", self._on_channel_destroyed)
        self._router.register("ChannelStateChange", self._on_channel_state_change)
        logger.info("call_controller_handlers_registered")

    async def _on_stasis_start(self, event: dict[str, Any]) -> None:
        """
        Handle incoming call entering Stasis application.

        Sets up the full call pipeline:
        1. Create session
        2. Answer channel
        3. Allocate RTP port and start bridge
        4. Create ExternalMedia channel
        5. Create mixing bridge and add both channels
        6. Start streaming pipeline
        """
        channel = event.get("channel", {})
        channel_id = channel.get("id", "")

        if not channel_id:
            logger.warning("stasis_start_no_channel_id", event=event)
            return

        # Check if this is an ExternalMedia channel (don't process those)
        args = event.get("args", [])
        if args and "external_media" in str(args).lower():
            return

        call_id = str(uuid.uuid4())
        set_call_id(call_id)

        logger.info(
            "stasis_start",
            call_id=call_id,
            channel_id=channel_id,
            caller=channel.get("caller", {}).get("number", "unknown"),
        )

        # Check concurrent call limit
        if self._shutting_down:
            logger.warning("call_rejected_shutting_down", call_id=call_id)
            await self._reject_call(channel_id)
            return

        if len(self._active_calls) >= self._settings.server.max_concurrent_calls:
            logger.warning(
                "call_rejected_max_concurrent",
                call_id=call_id,
                active=len(self._active_calls),
                limit=self._settings.server.max_concurrent_calls,
            )
            MetricsCollector.increment_call("rejected")
            await self._reject_call(channel_id)
            return

        try:
            # 1. Create session
            session = await self._session_mgr.create_session(call_id, channel_id)

            # 2. Answer channel
            await self._ari.answer_channel(channel_id)
            await self._session_mgr.transition_state(call_id, CallState.ANSWERED)

            # 3. Allocate RTP port and create bridge
            rtp_port = self._port_allocator.allocate()
            rtp_host = self._settings.rtp.listen_host
            if rtp_host == "0.0.0.0":
                rtp_host = "127.0.0.1"

            rtp_bridge = RTPBridge(
                call_id=call_id,
                local_port=rtp_port,
                settings=self._settings.rtp,
            )
            await rtp_bridge.start()

            # 4. Create ExternalMedia channel
            ext_media_response = await self._ari.create_external_media(
                external_host=f"{rtp_host}:{rtp_port}",
                fmt=self._settings.rtp.codec,
            )
            ext_media_channel_id = ext_media_response.get("channel", {}).get("id", "")
            if not ext_media_channel_id:
                ext_media_channel_id = ext_media_response.get("id", "")

            # 5. Create mixing bridge and add channels
            bridge_response = await self._ari.create_bridge(bridge_type="mixing")
            bridge_id = bridge_response.get("id", "")

            await self._ari.add_channel_to_bridge(bridge_id, channel_id)
            await self._ari.add_channel_to_bridge(bridge_id, ext_media_channel_id)

            # Update session with bridge info
            session.external_media_channel_id = ext_media_channel_id
            session.bridge_id = bridge_id
            session.rtp_port = rtp_port
            await self._session_mgr.update_session(session)

            # 6. Create and start streaming pipeline (via transport abstraction)
            from transport.ari_transport import ARITransport
            ari_transport = ARITransport(rtp_bridge)
            pipeline = StreamingPipeline(
                call_id=call_id,
                transport=ari_transport,
                asr_service=self._asr,
                llm_service=self._llm,
                tts_service=self._tts,
                vad_service=self._vad,
                session_manager=self._session_mgr,
            )
            await pipeline.start()

            # Transition to LISTENING
            await self._session_mgr.transition_state(call_id, CallState.LISTENING)

            # Register active call
            active_call = ActiveCall(
                call_id=call_id,
                channel_id=channel_id,
                pipeline=pipeline,
                rtp_bridge=rtp_bridge,
                bridge_id=bridge_id,
                external_media_channel_id=ext_media_channel_id,
            )

            async with self._lock:
                self._active_calls[channel_id] = active_call

            MetricsCollector.increment_call("answered")
            MetricsCollector.increment_active_calls()

            logger.info(
                "call_setup_complete",
                call_id=call_id,
                channel_id=channel_id,
                rtp_port=rtp_port,
                bridge_id=bridge_id,
            )

        except ResourceExhausted:
            logger.error("call_resource_exhausted", call_id=call_id)
            MetricsCollector.increment_call("error")
            await self._reject_call(channel_id)
        except (ARICommandError, SessionError, VoiceAssistantError) as exc:
            logger.exception("call_setup_failed", call_id=call_id, error=str(exc))
            MetricsCollector.increment_call("error")
            await self._cleanup_failed_setup(call_id, channel_id, locals())
        except Exception:
            logger.exception("call_setup_unexpected_error", call_id=call_id)
            MetricsCollector.increment_call("error")
            await self._cleanup_failed_setup(call_id, channel_id, locals())

    async def _on_stasis_end(self, event: dict[str, Any]) -> None:
        """Handle channel leaving Stasis."""
        channel = event.get("channel", {})
        channel_id = channel.get("id", "")
        await self._teardown_call(channel_id)

    async def _on_channel_destroyed(self, event: dict[str, Any]) -> None:
        """Handle channel destruction (hangup)."""
        channel = event.get("channel", {})
        channel_id = channel.get("id", "")
        await self._teardown_call(channel_id)

    async def _on_channel_state_change(self, event: dict[str, Any]) -> None:
        """Handle channel state changes."""
        channel = event.get("channel", {})
        channel_id = channel.get("id", "")
        state = channel.get("state", "")
        logger.debug(
            "channel_state_change",
            channel_id=channel_id,
            state=state,
        )

    async def _teardown_call(self, channel_id: str) -> None:
        """
        Tear down all resources for a call.

        Idempotent - safe to call multiple times.
        """
        async with self._lock:
            active = self._active_calls.pop(channel_id, None)

        if active is None:
            return

        call_id = active.call_id
        set_call_id(call_id)
        logger.info("call_teardown_begin", call_id=call_id, channel_id=channel_id)

        # Calculate call duration
        duration = time.monotonic() - active.start_time
        MetricsCollector.record_call_duration(duration)
        MetricsCollector.decrement_active_calls()

        # Stop pipeline
        try:
            await asyncio.wait_for(active.pipeline.stop(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("pipeline_stop_timeout", call_id=call_id)
        except Exception:
            logger.exception("pipeline_stop_error", call_id=call_id)

        # Stop RTP bridge
        try:
            await active.rtp_bridge.stop()
            self._port_allocator.release(active.rtp_bridge.local_port)
        except Exception:
            logger.exception("rtp_bridge_stop_error", call_id=call_id)

        # Destroy ARI bridge
        try:
            if active.bridge_id:
                await self._ari.destroy_bridge(active.bridge_id)
        except Exception:
            logger.exception("bridge_destroy_error", call_id=call_id)

        # Transition session to TERMINATED
        try:
            await self._session_mgr.transition_state(call_id, CallState.TERMINATED)
        except Exception:
            logger.exception("session_terminate_error", call_id=call_id)

        logger.info(
            "call_teardown_complete",
            call_id=call_id,
            duration_seconds=duration,
        )

    async def _reject_call(self, channel_id: str) -> None:
        """Reject an incoming call (busy/congestion)."""
        try:
            await self._ari.hangup_channel(channel_id, reason="congestion")
        except Exception:
            logger.exception("call_reject_error", channel_id=channel_id)

    async def _cleanup_failed_setup(
        self,
        call_id: str,
        channel_id: str,
        local_vars: dict[str, Any],
    ) -> None:
        """Clean up resources from a failed call setup."""
        # Hang up channel
        try:
            await self._ari.hangup_channel(channel_id, reason="normal")
        except Exception:
            pass

        # Release RTP port if allocated
        rtp_bridge = local_vars.get("rtp_bridge")
        if rtp_bridge and isinstance(rtp_bridge, RTPBridge):
            try:
                await rtp_bridge.stop()
                self._port_allocator.release(rtp_bridge.local_port)
            except Exception:
                pass

        # Destroy bridge if created
        bridge_id = local_vars.get("bridge_id", "")
        if bridge_id:
            try:
                await self._ari.destroy_bridge(bridge_id)
            except Exception:
                pass

        # Delete session if created
        try:
            await self._session_mgr.delete_session(call_id)
        except Exception:
            pass

    async def shutdown(self, drain_timeout: float = 30.0) -> None:
        """
        Gracefully shutdown the call controller.

        Stops accepting new calls and drains active calls.

        Args:
            drain_timeout: Maximum seconds to wait for active calls to complete.
        """
        self._shutting_down = True
        logger.info(
            "call_controller_shutdown_begin",
            active_calls=len(self._active_calls),
            drain_timeout=drain_timeout,
        )

        if not self._active_calls:
            logger.info("call_controller_shutdown_no_active_calls")
            return

        # Hang up all active calls
        async with self._lock:
            channel_ids = list(self._active_calls.keys())

        for channel_id in channel_ids:
            try:
                await self._ari.hangup_channel(channel_id, reason="normal")
            except Exception:
                logger.exception(
                    "shutdown_hangup_error",
                    channel_id=channel_id,
                )

        # Wait for all calls to teardown
        deadline = time.monotonic() + drain_timeout
        while self._active_calls and time.monotonic() < deadline:
            await asyncio.sleep(0.5)

        remaining = len(self._active_calls)
        if remaining > 0:
            logger.warning(
                "call_controller_shutdown_force",
                remaining_calls=remaining,
            )
            # Force teardown remaining calls
            async with self._lock:
                channel_ids = list(self._active_calls.keys())
            for channel_id in channel_ids:
                await self._teardown_call(channel_id)

        logger.info("call_controller_shutdown_complete")

    @property
    def active_call_count(self) -> int:
        return len(self._active_calls)

    def get_active_call(self, channel_id: str) -> ActiveCall | None:
        return self._active_calls.get(channel_id)
