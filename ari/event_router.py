"""
ARI event router with handler dispatch.

Routes ARI WebSocket events to registered handler coroutines by event type.
Error isolation ensures one handler failure doesn't affect others.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

from core.logger import get_logger
from core.metrics import ERRORS_TOTAL

if TYPE_CHECKING:
    from ari.ari_client import ARIClient

logger = get_logger(__name__)

EventHandler = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class EventRouter:
    """
    ARI event dispatcher with registered handler pattern.

    Receives events from the ARI WebSocket client and dispatches
    to registered async handlers by event type. Each handler runs
    in an isolated task to prevent cascading failures.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._running = False
        self._task: asyncio.Task | None = None

        logger.info("event_router_initialized")

    def register(self, event_type: str, handler: EventHandler) -> None:
        """
        Register a handler for a specific ARI event type.

        Args:
            event_type: ARI event type string (e.g., StasisStart).
            handler: Async callable that processes the event.
        """
        self._handlers[event_type].append(handler)
        logger.info(
            "event_handler_registered",
            event_type=event_type,
            handler=handler.__qualname__,
        )

    async def run(self, ari_client: ARIClient) -> None:
        """
        Start the event routing loop.

        Reads events from the ARI client and dispatches to handlers.

        Args:
            ari_client: Connected ARI client instance.
        """
        self._running = True
        logger.info("event_router_started")

        try:
            async for event in ari_client.get_events():
                if not self._running:
                    break
                await self._dispatch(event)
        except asyncio.CancelledError:
            logger.info("event_router_cancelled")
        except Exception:
            logger.exception("event_router_error")
        finally:
            self._running = False
            logger.info("event_router_stopped")

    async def stop(self) -> None:
        """Stop the event routing loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("event_router_stop_requested")

    async def _dispatch(self, event: dict[str, Any]) -> None:
        """
        Dispatch an event to all registered handlers.

        Each handler runs as an isolated asyncio task.
        Errors in one handler don't affect others.
        """
        event_type = event.get("type", "Unknown")
        handlers = self._handlers.get(event_type, [])

        if not handlers:
            logger.debug("event_no_handler", event_type=event_type)
            return

        logger.debug(
            "event_dispatch",
            event_type=event_type,
            handler_count=len(handlers),
        )

        for handler in handlers:
            asyncio.create_task(
                self._safe_handle(handler, event, event_type),
                name=f"handler_{event_type}_{handler.__qualname__}",
            )

    async def _safe_handle(
        self,
        handler: EventHandler,
        event: dict[str, Any],
        event_type: str,
    ) -> None:
        """Execute a handler with error isolation."""
        try:
            await handler(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            ERRORS_TOTAL.labels(category="ari").inc()
            logger.exception(
                "event_handler_error",
                event_type=event_type,
                handler=handler.__qualname__,
            )
