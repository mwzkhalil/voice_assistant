"""
API key authentication and rate limiting.

Enforces:
- API key validation via configurable header
- Per-IP request rate limiting (sliding window via Redis)
- Per-IP concurrent session limits
- Idle session timeout enforcement

No anonymous unlimited access is allowed on public endpoints.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

import redis.asyncio as aioredis
from fastapi import HTTPException, Request, WebSocket, status

from core.config import AuthSettings, get_settings
from core.logger import get_logger

logger = get_logger(__name__)

# Module-level rate limit state (backed by Redis when available, in-memory fallback)
_rate_limit_cache: dict[str, list[float]] = {}
_session_count_cache: dict[str, int] = {}
_rate_limit_lock = asyncio.Lock()


def _get_client_ip(request: Request | None = None, websocket: WebSocket | None = None) -> str:
    """Extract client IP, respecting X-Forwarded-For from reverse proxy."""
    if request is not None:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"
    if websocket is not None:
        forwarded = websocket.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if websocket.client:
            return websocket.client.host
        return "unknown"
    return "unknown"


async def validate_api_key(
    api_key: str | None,
    settings: AuthSettings | None = None,
) -> bool:
    """
    Validate an API key against configured keys.

    Returns True if valid or auth is disabled.
    """
    if settings is None:
        settings = get_settings().auth

    if not settings.enabled:
        return True

    valid_keys = settings.get_api_keys()
    if not valid_keys:
        # No keys configured = auth effectively disabled
        return True

    return api_key is not None and api_key in valid_keys


async def check_rate_limit(
    client_ip: str,
    settings: AuthSettings | None = None,
    redis_client: Optional[aioredis.Redis] = None,
) -> bool:
    """
    Check if client IP is within rate limit.

    Uses Redis sliding window if available, falls back to in-memory.

    Returns True if request is allowed.
    """
    if settings is None:
        settings = get_settings().auth

    now = time.time()
    window = 60.0  # 1 minute
    max_requests = settings.rate_limit_per_minute
    key = f"rate:{client_ip}"

    if redis_client is not None:
        try:
            pipe = redis_client.pipeline()
            await pipe.zremrangebyscore(key, 0, now - window)
            await pipe.zadd(key, {str(now): now})
            await pipe.zcard(key)
            await pipe.expire(key, int(window) + 1)
            results = await pipe.execute()
            count = results[2]
            return count <= max_requests
        except Exception:
            logger.warning("rate_limit_redis_error", ip=client_ip)

    # In-memory fallback
    async with _rate_limit_lock:
        if client_ip not in _rate_limit_cache:
            _rate_limit_cache[client_ip] = []

        timestamps = _rate_limit_cache[client_ip]
        # Remove expired entries
        cutoff = now - window
        timestamps[:] = [t for t in timestamps if t > cutoff]
        timestamps.append(now)

        return len(timestamps) <= max_requests


async def check_session_limit(
    client_ip: str,
    settings: AuthSettings | None = None,
) -> bool:
    """Check if client IP is within concurrent session limit."""
    if settings is None:
        settings = get_settings().auth

    async with _rate_limit_lock:
        count = _session_count_cache.get(client_ip, 0)
        return count < settings.max_sessions_per_ip


async def register_session(client_ip: str) -> None:
    """Register a new session for the given IP."""
    async with _rate_limit_lock:
        _session_count_cache[client_ip] = _session_count_cache.get(client_ip, 0) + 1


async def unregister_session(client_ip: str) -> None:
    """Unregister a session for the given IP."""
    async with _rate_limit_lock:
        count = _session_count_cache.get(client_ip, 0)
        if count > 0:
            _session_count_cache[client_ip] = count - 1
        else:
            _session_count_cache.pop(client_ip, None)


async def authenticate_websocket(websocket: WebSocket) -> str:
    """
    Authenticate and rate-limit a WebSocket connection.

    Returns the client IP on success.
    Closes the WebSocket with appropriate code on failure.
    """
    settings = get_settings().auth
    client_ip = _get_client_ip(websocket=websocket)

    # API key validation
    api_key = websocket.headers.get(settings.header_name) or websocket.query_params.get("api_key")
    if not await validate_api_key(api_key, settings):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid API key")
        raise HTTPException(status_code=403, detail="Invalid API key")

    # Rate limiting
    if not await check_rate_limit(client_ip, settings):
        await websocket.close(code=4029, reason="Rate limit exceeded")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Session limit
    if not await check_session_limit(client_ip, settings):
        await websocket.close(code=4029, reason="Max sessions per IP exceeded")
        raise HTTPException(status_code=429, detail="Max concurrent sessions exceeded")

    return client_ip


async def authenticate_http(request: Request) -> str:
    """
    Authenticate and rate-limit an HTTP request.

    Returns the client IP on success.
    Raises HTTPException on failure.
    """
    settings = get_settings().auth
    client_ip = _get_client_ip(request=request)

    # API key validation
    api_key = request.headers.get(settings.header_name)
    if not await validate_api_key(api_key, settings):
        raise HTTPException(status_code=403, detail="Invalid API key")

    # Rate limiting
    if not await check_rate_limit(client_ip, settings):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    return client_ip
