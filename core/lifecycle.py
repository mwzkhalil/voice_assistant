"""
Application lifecycle management.

Ordered startup and graceful shutdown of all services.
Integrates with FastAPI lifespan context manager.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Any

from core.logger import get_logger

logger = get_logger(__name__)

StartupHook = Callable[[], Coroutine[Any, Any, None]]
ShutdownHook = Callable[[], Coroutine[Any, Any, None]]


class LifecycleManager:
    """
    Manages ordered application startup and graceful shutdown.

    Startup hooks execute in registration order (FIFO).
    Shutdown hooks execute in reverse registration order (LIFO)
    to ensure clean dependency teardown.
    """

    def __init__(self) -> None:
        self._startup_hooks: list[tuple[str, StartupHook]] = []
        self._shutdown_hooks: list[tuple[str, ShutdownHook]] = []
        self._started: bool = False

    def register_startup(self, name: str, hook: StartupHook) -> None:
        """Register an async startup hook with a descriptive name."""
        self._startup_hooks.append((name, hook))

    def register_shutdown(self, name: str, hook: ShutdownHook) -> None:
        """Register an async shutdown hook with a descriptive name."""
        self._shutdown_hooks.append((name, hook))

    async def startup(self) -> None:
        """Execute all startup hooks in registration order."""
        logger.info("lifecycle_startup_begin", hook_count=len(self._startup_hooks))

        for name, hook in self._startup_hooks:
            logger.info("lifecycle_startup_hook", hook_name=name)
            try:
                await hook()
                logger.info("lifecycle_startup_hook_complete", hook_name=name)
            except Exception:
                logger.exception("lifecycle_startup_hook_failed", hook_name=name)
                raise

        self._started = True
        logger.info("lifecycle_startup_complete")

    async def shutdown(self, drain_timeout: float = 30.0) -> None:
        """
        Execute all shutdown hooks in reverse registration order.

        Args:
            drain_timeout: Maximum seconds to wait for graceful drain.
        """
        logger.info(
            "lifecycle_shutdown_begin",
            hook_count=len(self._shutdown_hooks),
            drain_timeout=drain_timeout,
        )

        # Execute shutdown hooks in reverse order (LIFO)
        for name, hook in reversed(self._shutdown_hooks):
            logger.info("lifecycle_shutdown_hook", hook_name=name)
            try:
                await asyncio.wait_for(hook(), timeout=drain_timeout)
                logger.info("lifecycle_shutdown_hook_complete", hook_name=name)
            except asyncio.TimeoutError:
                logger.error(
                    "lifecycle_shutdown_hook_timeout",
                    hook_name=name,
                    timeout=drain_timeout,
                )
            except Exception:
                logger.exception("lifecycle_shutdown_hook_failed", hook_name=name)

        self._started = False
        logger.info("lifecycle_shutdown_complete")

    @property
    def is_started(self) -> bool:
        return self._started


# Module-level singleton
_lifecycle: LifecycleManager | None = None


def get_lifecycle() -> LifecycleManager:
    """Return the singleton lifecycle manager."""
    global _lifecycle
    if _lifecycle is None:
        _lifecycle = LifecycleManager()
    return _lifecycle


def reset_lifecycle() -> None:
    """Reset lifecycle singleton (for testing)."""
    global _lifecycle
    _lifecycle = None


@asynccontextmanager
async def app_lifespan(app: Any) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Wires up service container initialization and teardown.
    """
    from core.dependency_injection import get_container

    lifecycle = get_lifecycle()
    container = get_container()

    # Register container startup / shutdown
    lifecycle.register_startup("service_container", container.initialize_all)
    lifecycle.register_shutdown("service_container", container.shutdown_all)

    await lifecycle.startup()
    try:
        yield
    finally:
        await lifecycle.shutdown()
