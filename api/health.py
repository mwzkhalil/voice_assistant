"""
Health and readiness endpoints.

GET /health  - Liveness probe (always 200 if process running)
GET /ready   - Readiness probe (checks Redis, models, transport-specific deps)

Transport-mode aware: ARI checks only run in ARI mode.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response, status

from core.config import TransportMode
from core.dependency_injection import get_container
from core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    response_model=dict[str, Any],
)
async def health() -> dict[str, Any]:
    """
    Liveness check. Returns 200 if the process is running.
    """
    container = get_container()
    return {
        "status": "alive",
        "service": "voice-assistant",
        "transport_mode": container.transport_mode.value,
    }


@router.get(
    "/ready",
    summary="Readiness probe",
    response_model=dict[str, Any],
)
async def readiness(response: Response) -> dict[str, Any]:
    """
    Readiness check. Verifies all critical dependencies are operational.

    Returns 200 if ready, 503 if not ready.
    Transport-mode aware: ARI checks only run in ARI mode.
    """
    container = get_container()
    checks: dict[str, dict[str, Any]] = {}
    all_ready = True

    # Check service container initialization
    if not container.is_initialized:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "not_ready",
            "reason": "Service container not initialized",
            "checks": {},
        }

    # Redis check
    try:
        redis_ok = await container.session_manager.health_check()
        checks["redis"] = {"status": "ok" if redis_ok else "error"}
        if not redis_ok:
            all_ready = False
    except Exception as exc:
        checks["redis"] = {"status": "error", "detail": str(exc)}
        all_ready = False

    # ARI connection check (only in ARI mode)
    if container.transport_mode == TransportMode.ARI:
        try:
            ari = container.ari_client
            ari_ok = ari is not None and ari.is_connected
            checks["ari"] = {"status": "ok" if ari_ok else "disconnected"}
            if not ari_ok:
                all_ready = False
        except Exception as exc:
            checks["ari"] = {"status": "error", "detail": str(exc)}
            all_ready = False
    else:
        checks["transport"] = {"mode": "websocket", "status": "ok"}

    # ASR model check
    try:
        asr_ok = container.asr_service.is_loaded
        checks["asr_model"] = {"status": "loaded" if asr_ok else "not_loaded"}
        if not asr_ok:
            all_ready = False
    except Exception as exc:
        checks["asr_model"] = {"status": "error", "detail": str(exc)}
        all_ready = False

    # TTS model check
    try:
        tts_ok = container.tts_service.is_loaded
        checks["tts_model"] = {"status": "loaded" if tts_ok else "not_loaded"}
        if not tts_ok:
            all_ready = False
    except Exception as exc:
        checks["tts_model"] = {"status": "error", "detail": str(exc)}
        all_ready = False

    # Active sessions info
    if container.transport_mode == TransportMode.ARI:
        try:
            controller = container.call_controller
            active = controller.active_call_count if controller else 0
            checks["calls"] = {"active": active}
        except Exception:
            checks["calls"] = {"active": 0}
    else:
        from api.realtime import get_active_ws_session_count
        try:
            active = await get_active_ws_session_count()
            checks["sessions"] = {"active_websocket": active}
        except Exception:
            checks["sessions"] = {"active_websocket": 0}

    if not all_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ready" if all_ready else "not_ready",
        "transport_mode": container.transport_mode.value,
        "checks": checks,
    }
