"""
Dependency injection container.

Provides lazy-initialized singleton access to all services.
Integrates with FastAPI Depends().
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Optional

from core.config import TransportMode, get_settings
from core.logger import configure_logging, get_logger
from core.metrics import APP_INFO, MetricsCollector

if TYPE_CHECKING:
    from ari.ari_client import ARIClient
    from ari.call_controller import CallController
    from ari.event_router import EventRouter
    from services.asr_service import ASRService
    from services.llm_service import LLMService
    from services.session_manager import SessionManager
    from services.tts_service import TTSService
    from services.vad_service import VADService

logger = get_logger(__name__)


class ServiceContainer:
    """
    Centralized service container with lazy initialization.

    All services are created once and shared across the application.
    Initialization and shutdown are ordered to respect dependencies.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._redis_pool: Any = None
        self._asr_service: Optional[ASRService] = None
        self._tts_service: Optional[TTSService] = None
        self._llm_service: Optional[LLMService] = None
        self._vad_service: Optional[VADService] = None
        self._session_manager: Optional[SessionManager] = None
        self._ari_client: Optional[ARIClient] = None
        self._event_router: Optional[EventRouter] = None
        self._call_controller: Optional[CallController] = None
        self._metrics: MetricsCollector = MetricsCollector()
        self._initialized: bool = False
        self._lock = asyncio.Lock()

    async def initialize_all(self) -> None:
        """Initialize all services in dependency order."""
        async with self._lock:
            if self._initialized:
                return

            settings = self._settings
            configure_logging(settings.server.log_level)

            APP_INFO.info({
                "version": "1.0.0",
                "llm_model": settings.ollama.model_name,
                "asr_model": settings.asr.model_name,
                "tts_model": settings.tts.model_name,
            })

            transport_mode = settings.server.transport_mode
            logger.info("container_init_begin", transport_mode=transport_mode.value)

            # 1. Redis connection pool
            await self._init_redis()

            # 2. Session manager (depends on Redis)
            await self._init_session_manager()

            # 3. VAD service (no dependencies)
            await self._init_vad()

            # 4. ASR service (depends on GPU)
            await self._init_asr()

            # 5. TTS service (depends on GPU)
            await self._init_tts()

            # 6. LLM service (depends on network)
            await self._init_llm()

            # 7-8. ARI client + call controller (only in ARI mode)
            if transport_mode == TransportMode.ARI:
                await self._init_ari()
                await self._init_call_controller()
            else:
                logger.info("ari_skipped_ws_mode")

            self._initialized = True
            logger.info("container_init_complete", transport_mode=transport_mode.value)

    async def _init_redis(self) -> None:
        """Initialize Redis connection pool."""
        import redis.asyncio as aioredis

        logger.info("init_redis", url=self._settings.redis.url)
        self._redis_pool = aioredis.ConnectionPool.from_url(
            self._settings.redis.url,
            max_connections=self._settings.redis.pool_max_size,
            socket_timeout=self._settings.redis.socket_timeout,
            socket_connect_timeout=self._settings.redis.socket_connect_timeout,
            retry_on_timeout=self._settings.redis.retry_on_timeout,
            health_check_interval=self._settings.redis.health_check_interval,
            decode_responses=True,
        )
        # Verify connection
        client = aioredis.Redis(connection_pool=self._redis_pool)
        await client.ping()
        await client.aclose()
        logger.info("init_redis_complete")

    async def _init_session_manager(self) -> None:
        """Initialize session manager."""
        from services.session_manager import SessionManager

        logger.info("init_session_manager")
        self._session_manager = SessionManager(
            redis_pool=self._redis_pool,
            settings=self._settings.redis,
        )
        logger.info("init_session_manager_complete")

    async def _init_vad(self) -> None:
        """Initialize Voice Activity Detection service."""
        from services.vad_service import VADService

        logger.info("init_vad")
        self._vad_service = VADService(settings=self._settings.vad)
        logger.info("init_vad_complete")

    async def _init_asr(self) -> None:
        """Initialize ASR service (loads model to GPU)."""
        from services.asr_service import ASRService

        logger.info("init_asr", model=self._settings.asr.model_name)
        self._asr_service = ASRService(settings=self._settings.asr)
        await self._asr_service.load_model()
        logger.info("init_asr_complete")

    async def _init_tts(self) -> None:
        """Initialize TTS service (loads model to GPU)."""
        from services.tts_service import TTSService

        logger.info("init_tts", model=self._settings.tts.model_name)
        self._tts_service = TTSService(settings=self._settings.tts)
        await self._tts_service.load_model()
        logger.info("init_tts_complete")

    async def _init_llm(self) -> None:
        """Initialize LLM service."""
        from services.llm_service import LLMService

        logger.info("init_llm", model=self._settings.ollama.model_name)
        self._llm_service = LLMService(settings=self._settings.ollama)
        logger.info("init_llm_complete")

    async def _init_ari(self) -> None:
        """Initialize ARI WebSocket client."""
        from ari.ari_client import ARIClient

        logger.info("init_ari", host=self._settings.ari.host)
        self._ari_client = ARIClient(settings=self._settings.ari)
        await self._ari_client.connect()
        logger.info("init_ari_complete")

    async def _init_call_controller(self) -> None:
        """Initialize event router and call controller."""
        from ari.call_controller import CallController
        from ari.event_router import EventRouter

        logger.info("init_call_controller")
        assert self._ari_client is not None
        assert self._session_manager is not None
        assert self._asr_service is not None
        assert self._tts_service is not None
        assert self._llm_service is not None
        assert self._vad_service is not None

        self._event_router = EventRouter()
        self._call_controller = CallController(
            ari_client=self._ari_client,
            event_router=self._event_router,
            session_manager=self._session_manager,
            asr_service=self._asr_service,
            tts_service=self._tts_service,
            llm_service=self._llm_service,
            vad_service=self._vad_service,
            settings=self._settings,
        )
        self._call_controller.register_handlers()
        # Start event processing loop
        asyncio.create_task(self._event_router.run(self._ari_client))
        logger.info("init_call_controller_complete")

    async def shutdown_all(self) -> None:
        """Shutdown all services in reverse dependency order."""
        async with self._lock:
            logger.info("container_shutdown_begin")

            # 1. Stop call controller (drains active calls)
            if self._call_controller is not None:
                await self._call_controller.shutdown()

            # 2. Disconnect ARI
            if self._ari_client is not None:
                await self._ari_client.disconnect()

            # 3. Unload models
            if self._asr_service is not None:
                await self._asr_service.unload_model()
            if self._tts_service is not None:
                await self._tts_service.unload_model()

            # 4. Close LLM client
            if self._llm_service is not None:
                await self._llm_service.close()

            # 5. Close Redis pool
            if self._redis_pool is not None:
                await self._redis_pool.aclose()

            self._initialized = False
            logger.info("container_shutdown_complete")

    # --- Typed Getters ---

    @property
    def redis_pool(self) -> Any:
        assert self._redis_pool is not None, "Redis pool not initialized"
        return self._redis_pool

    @property
    def asr_service(self) -> ASRService:
        assert self._asr_service is not None, "ASR service not initialized"
        return self._asr_service

    @property
    def tts_service(self) -> TTSService:
        assert self._tts_service is not None, "TTS service not initialized"
        return self._tts_service

    @property
    def llm_service(self) -> LLMService:
        assert self._llm_service is not None, "LLM service not initialized"
        return self._llm_service

    @property
    def vad_service(self) -> VADService:
        assert self._vad_service is not None, "VAD service not initialized"
        return self._vad_service

    @property
    def session_manager(self) -> SessionManager:
        assert self._session_manager is not None, "Session manager not initialized"
        return self._session_manager

    @property
    def ari_client(self) -> ARIClient | None:
        return self._ari_client

    @property
    def event_router(self) -> EventRouter | None:
        return self._event_router

    @property
    def call_controller(self) -> CallController | None:
        return self._call_controller

    @property
    def transport_mode(self) -> TransportMode:
        return self._settings.server.transport_mode

    @property
    def metrics(self) -> MetricsCollector:
        return self._metrics

    @property
    def is_initialized(self) -> bool:
        return self._initialized


# Module-level singleton
_container: ServiceContainer | None = None


def get_container() -> ServiceContainer:
    """Return the singleton service container."""
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container


def reset_container() -> None:
    """Reset the container singleton (for testing)."""
    global _container
    _container = None
