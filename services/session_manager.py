"""
Redis-backed call session manager.

Enforces a strict state machine for call lifecycle transitions.
Distributed locking via Redis ensures serialized state updates.
Sessions are persisted with TTL for automatic cleanup.
"""

from __future__ import annotations

import enum
import json
import time
from typing import Any, Optional

import redis.asyncio as aioredis
from pydantic import BaseModel, Field

from core.config import RedisSettings
from core.exceptions import InvalidStateTransition, SessionError
from core.logger import get_logger
from core.metrics import MetricsCollector

logger = get_logger(__name__)


class CallState(str, enum.Enum):
    """Strict call session states with enforced transitions."""
    INIT = "INIT"
    ANSWERED = "ANSWERED"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"
    TERMINATED = "TERMINATED"


# Valid state transition matrix
VALID_TRANSITIONS: dict[CallState, set[CallState]] = {
    CallState.INIT: {CallState.ANSWERED, CallState.TERMINATED},
    CallState.ANSWERED: {CallState.LISTENING, CallState.TERMINATED},
    CallState.LISTENING: {CallState.THINKING, CallState.TERMINATED},
    CallState.THINKING: {CallState.SPEAKING, CallState.TERMINATED},
    CallState.SPEAKING: {CallState.INTERRUPTED, CallState.LISTENING, CallState.TERMINATED},
    CallState.INTERRUPTED: {CallState.LISTENING, CallState.TERMINATED},
    CallState.TERMINATED: set(),  # Terminal state
}


class ConversationMessage(BaseModel):
    """A single message in the conversation history."""
    role: str
    content: str
    timestamp: float = Field(default_factory=time.time)


class CallSession(BaseModel):
    """
    Complete call session state persisted to Redis.

    Every active call has exactly one CallSession.
    """
    call_id: str
    channel_id: str
    state: CallState = CallState.INIT
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    answered_at: Optional[float] = None
    terminated_at: Optional[float] = None
    conversation_history: list[ConversationMessage] = Field(default_factory=list)
    interrupt_flag: bool = False
    rate_limit_tokens: int = 0
    rate_limit_reset_at: float = 0.0
    external_media_channel_id: Optional[str] = None
    bridge_id: Optional[str] = None
    rtp_port: Optional[int] = None
    remote_rtp_host: Optional[str] = None
    remote_rtp_port: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_history_for_llm(self) -> list[dict[str, str]]:
        """Return conversation history formatted for LLM input."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.conversation_history
        ]

    def add_message(self, role: str, content: str) -> None:
        """Append a message to conversation history."""
        self.conversation_history.append(
            ConversationMessage(role=role, content=content)
        )
        self.updated_at = time.time()


class SessionManager:
    """
     session manager with Redis persistence and distributed locking.

    Manages call session lifecycle with strict state machine enforcement.
    All state transitions are validated and locked to prevent race conditions.
    """

    SESSION_PREFIX = "session:"
    LOCK_PREFIX = "lock:session:"

    def __init__(
        self,
        redis_pool: aioredis.ConnectionPool,
        settings: RedisSettings,
    ) -> None:
        self._pool = redis_pool
        self._settings = settings
        self._client = aioredis.Redis(connection_pool=redis_pool)

        logger.info(
            "session_manager_initialized",
            session_ttl=settings.session_ttl,
            lock_timeout=settings.lock_timeout,
        )

    def _session_key(self, call_id: str) -> str:
        return f"{self.SESSION_PREFIX}{call_id}"

    def _lock_key(self, call_id: str) -> str:
        return f"{self.LOCK_PREFIX}{call_id}"

    async def create_session(self, call_id: str, channel_id: str) -> CallSession:
        """
        Create a new call session in INIT state.

        Args:
            call_id: Unique call identifier.
            channel_id: Asterisk channel ID.

        Returns:
            The newly created CallSession.

        Raises:
            SessionError: If session already exists.
        """
        key = self._session_key(call_id)

        # Atomic check-and-set
        exists = await self._client.exists(key)
        if exists:
            raise SessionError(
                f"Session already exists for call_id={call_id}",
                call_id=call_id,
            )

        session = CallSession(call_id=call_id, channel_id=channel_id)
        await self._client.set(
            key,
            session.model_dump_json(),
            ex=self._settings.session_ttl,
        )

        logger.info("session_created", call_id=call_id, channel_id=channel_id)
        return session

    async def get_session(self, call_id: str) -> CallSession:
        """
        Retrieve a call session.

        Args:
            call_id: Unique call identifier.

        Returns:
            The CallSession.

        Raises:
            SessionError: If session not found.
        """
        key = self._session_key(call_id)
        data = await self._client.get(key)
        if data is None:
            raise SessionError(f"Session not found: {call_id}", call_id=call_id)

        return CallSession.model_validate_json(data)

    async def update_session(self, session: CallSession) -> None:
        """
        Persist updated session data to Redis.

        Args:
            session: The session to persist.
        """
        key = self._session_key(session.call_id)
        session.updated_at = time.time()
        await self._client.set(
            key,
            session.model_dump_json(),
            ex=self._settings.session_ttl,
        )

    async def transition_state(
        self,
        call_id: str,
        target_state: CallState,
    ) -> CallSession:
        """
        Atomically transition a session to a new state with validation.

        Uses Redis distributed locking to ensure serialized access.

        Args:
            call_id: Unique call identifier.
            target_state: Desired target state.

        Returns:
            Updated CallSession.

        Raises:
            InvalidStateTransition: If transition is not valid.
            SessionError: If session not found or lock acquisition fails.
        """
        lock_key = self._lock_key(call_id)
        lock = self._client.lock(
            lock_key,
            timeout=self._settings.lock_timeout,
            blocking_timeout=self._settings.lock_timeout,
        )

        acquired = await lock.acquire()
        if not acquired:
            raise SessionError(
                f"Failed to acquire lock for session {call_id}",
                call_id=call_id,
            )

        try:
            session = await self.get_session(call_id)

            # Validate transition
            valid_targets = VALID_TRANSITIONS.get(session.state, set())
            if target_state not in valid_targets:
                raise InvalidStateTransition(
                    current_state=session.state.value,
                    target_state=target_state.value,
                    call_id=call_id,
                )

            old_state = session.state
            session.state = target_state
            session.updated_at = time.time()

            # Set lifecycle timestamps
            if target_state == CallState.ANSWERED:
                session.answered_at = time.time()
            elif target_state == CallState.TERMINATED:
                session.terminated_at = time.time()
            elif target_state == CallState.INTERRUPTED:
                session.interrupt_flag = True
            elif target_state == CallState.LISTENING:
                session.interrupt_flag = False

            await self.update_session(session)

            logger.info(
                "session_state_transition",
                call_id=call_id,
                from_state=old_state.value,
                to_state=target_state.value,
            )

            return session

        finally:
            try:
                await lock.release()
            except Exception:
                logger.warning("session_lock_release_failed", call_id=call_id)

    async def delete_session(self, call_id: str) -> None:
        """
        Delete a session from Redis.

        Args:
            call_id: Unique call identifier.
        """
        key = self._session_key(call_id)
        lock_key = self._lock_key(call_id)
        await self._client.delete(key, lock_key)
        logger.info("session_deleted", call_id=call_id)

    async def get_active_session_count(self) -> int:
        """Return count of active (non-terminated) sessions."""
        cursor = 0
        count = 0
        while True:
            cursor, keys = await self._client.scan(
                cursor=cursor,
                match=f"{self.SESSION_PREFIX}*",
                count=100,
            )
            for key in keys:
                data = await self._client.get(key)
                if data:
                    session = CallSession.model_validate_json(data)
                    if session.state != CallState.TERMINATED:
                        count += 1
            if cursor == 0:
                break
        return count

    async def cleanup_stale_sessions(self, max_age_seconds: float = 3600) -> int:
        """
        Clean up sessions that have been inactive for too long.

        Args:
            max_age_seconds: Maximum session age in seconds.

        Returns:
            Number of sessions cleaned up.
        """
        cursor = 0
        cleaned = 0
        now = time.time()

        while True:
            cursor, keys = await self._client.scan(
                cursor=cursor,
                match=f"{self.SESSION_PREFIX}*",
                count=100,
            )
            for key in keys:
                data = await self._client.get(key)
                if data:
                    session = CallSession.model_validate_json(data)
                    if now - session.updated_at > max_age_seconds:
                        await self._client.delete(key)
                        cleaned += 1
                        logger.info(
                            "session_stale_cleaned",
                            call_id=session.call_id,
                            age_seconds=now - session.updated_at,
                        )
            if cursor == 0:
                break

        return cleaned

    async def health_check(self) -> bool:
        """Check Redis connectivity."""
        try:
            result = await self._client.ping()
            return bool(result)
        except Exception:
            return False
