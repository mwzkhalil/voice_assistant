"""
FastAPI application entrypoint.

Application factory with lifespan management, middleware,
exception handlers, and router registration.
"""

from __future__ import annotations

import sys
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import get_settings
from core.exceptions import (
    CircuitBreakerOpen,
    ResourceExhausted,
    VoiceAssistantError,
)
from core.lifecycle import app_lifespan
from core.logger import get_logger, set_correlation_id

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """
    Application factory.

    Creates and configures the FastAPI application with all
    middleware, exception handlers, and routers.
    """
    settings = get_settings()

    app = FastAPI(
        title="Voice Assistant (Urdu)",
        description=(
            "Enterprise-grade Urdu voice assistant with Asterisk ARI integration, "
            "real-time ASR/LLM/TTS streaming pipeline."
        ),
        version="1.0.0",
        lifespan=app_lifespan,
        docs_url="/docs" if settings.server.debug else None,
        redoc_url="/redoc" if settings.server.debug else None,
    )

    # --- CORS Middleware ---
    origins = [o.strip() for o in settings.server.cors_origins.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Request ID Middleware ---
    @app.middleware("http")
    async def correlation_id_middleware(
        request: Request, call_next: Any
    ) -> Response:
        """Inject correlation ID into every request context."""
        cid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        set_correlation_id(cid)
        response = await call_next(request)
        response.headers["X-Request-ID"] = cid
        return response

    # --- Exception Handlers ---
    @app.exception_handler(VoiceAssistantError)
    async def voice_assistant_error_handler(
        request: Request, exc: VoiceAssistantError
    ) -> JSONResponse:
        """Handle all application-specific errors."""
        logger.error(
            "application_error",
            error_type=exc.__class__.__name__,
            message=str(exc),
            context=exc.context,
        )
        status_code = 500
        if isinstance(exc, ResourceExhausted):
            status_code = 503
        elif isinstance(exc, CircuitBreakerOpen):
            status_code = 503
        return JSONResponse(
            status_code=status_code,
            content=exc.to_dict(),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all for unhandled exceptions."""
        logger.exception(
            "unhandled_error",
            path=str(request.url),
            method=request.method,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error_type": "InternalServerError",
                "message": "An unexpected error occurred",
            },
        )

    # --- Static Files (frontend test client) ---
    import os
    from fastapi.staticfiles import StaticFiles

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # --- Register Routers ---
    from api.health import router as health_router
    from api.metrics_endpoint import router as metrics_router
    from api.websocket_debug import router as debug_router
    from api.realtime import router as realtime_router

    app.include_router(health_router)
    app.include_router(metrics_router)
    app.include_router(debug_router)
    app.include_router(realtime_router)

    return app


# Module-level app instance for Uvicorn
app = create_app()


def main() -> None:
    """Run the application with Uvicorn."""
    settings = get_settings()

    uvicorn.run(
        "main:app",
        host=settings.server.host,
        port=settings.server.port,
        workers=settings.server.workers,
        log_level=settings.server.log_level.value.lower(),
        access_log=False,  # We use structlog instead
        loop="uvloop" if sys.platform != "win32" else "asyncio",
        http="httptools" if sys.platform != "win32" else "h11",
        ws="websockets",
        timeout_keep_alive=30,
        limit_concurrency=1000,
        limit_max_requests=None,
    )


if __name__ == "__main__":
    main()
