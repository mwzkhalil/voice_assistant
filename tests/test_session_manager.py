"""
Tests for the call session manager.

Covers state machine transitions, invalid transition rejection,
session CRUD, and distributed locking.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from core.config import RedisSettings
from core.exceptions import InvalidStateTransition, SessionError
from services.session_manager import (
    CallSession,
    CallState,
    SessionManager,
    VALID_TRANSITIONS,
)


@pytest.fixture
def redis_settings() -> RedisSettings:
    return RedisSettings(
        url="redis://localhost:6379/15",
        session_ttl=60,
        lock_timeout=5.0,
    )


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client with lock support."""
    client = AsyncMock()
    client.ping.return_value = True
    client.exists.return_value = False
    client.delete.return_value = 1

    # Mock lock
    lock_mock = AsyncMock()
    lock_mock.acquire.return_value = True
    lock_mock.release.return_value = None
    lock_mock.__aenter__ = AsyncMock(return_value=lock_mock)
    lock_mock.__aexit__ = AsyncMock(return_value=None)
    client.lock.return_value = lock_mock

    return client


@pytest.fixture
def session_manager(mock_redis_client, redis_settings):
    """Create a SessionManager with mocked Redis."""
    pool = MagicMock()
    mgr = SessionManager(redis_pool=pool, settings=redis_settings)
    mgr._client = mock_redis_client
    return mgr


class TestCallSessionModel:
    """Tests for the CallSession Pydantic model."""

    def test_default_state(self):
        session = CallSession(call_id="test-1", channel_id="ch-1")
        assert session.state == CallState.INIT
        assert session.call_id == "test-1"
        assert session.channel_id == "ch-1"
        assert session.conversation_history == []
        assert session.interrupt_flag is False

    def test_add_message(self):
        session = CallSession(call_id="test-1", channel_id="ch-1")
        session.add_message("user", "ہیلو")
        session.add_message("assistant", "خوش آمدید")
        assert len(session.conversation_history) == 2
        assert session.conversation_history[0].role == "user"
        assert session.conversation_history[1].content == "خوش آمدید"

    def test_get_history_for_llm(self):
        session = CallSession(call_id="test-1", channel_id="ch-1")
        session.add_message("user", "سلام")
        session.add_message("assistant", "وعلیکم السلام")
        history = session.get_history_for_llm()
        assert history == [
            {"role": "user", "content": "سلام"},
            {"role": "assistant", "content": "وعلیکم السلام"},
        ]

    def test_serialization(self):
        session = CallSession(call_id="test-1", channel_id="ch-1")
        session.add_message("user", "ٹیسٹ")
        json_str = session.model_dump_json()
        restored = CallSession.model_validate_json(json_str)
        assert restored.call_id == "test-1"
        assert len(restored.conversation_history) == 1


class TestValidTransitions:
    """Tests for the state transition matrix."""

    def test_all_states_have_entries(self):
        for state in CallState:
            assert state in VALID_TRANSITIONS

    def test_terminated_is_terminal(self):
        assert VALID_TRANSITIONS[CallState.TERMINATED] == set()

    def test_init_transitions(self):
        assert VALID_TRANSITIONS[CallState.INIT] == {
            CallState.ANSWERED, CallState.TERMINATED
        }

    def test_speaking_transitions(self):
        assert VALID_TRANSITIONS[CallState.SPEAKING] == {
            CallState.INTERRUPTED, CallState.LISTENING, CallState.TERMINATED
        }

    def test_every_state_can_reach_terminated(self):
        for state in CallState:
            if state != CallState.TERMINATED:
                assert CallState.TERMINATED in VALID_TRANSITIONS[state], \
                    f"{state} cannot transition to TERMINATED"


class TestSessionManager:
    """Tests for SessionManager operations."""

    @pytest.mark.asyncio
    async def test_create_session(self, session_manager, mock_redis_client):
        mock_redis_client.exists.return_value = False

        session = await session_manager.create_session("call-1", "ch-1")
        assert session.call_id == "call-1"
        assert session.channel_id == "ch-1"
        assert session.state == CallState.INIT
        mock_redis_client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_duplicate_session_raises(self, session_manager, mock_redis_client):
        mock_redis_client.exists.return_value = True

        with pytest.raises(SessionError, match="already exists"):
            await session_manager.create_session("call-1", "ch-1")

    @pytest.mark.asyncio
    async def test_get_session(self, session_manager, mock_redis_client):
        session = CallSession(call_id="call-1", channel_id="ch-1")
        mock_redis_client.get.return_value = session.model_dump_json()

        result = await session_manager.get_session("call-1")
        assert result.call_id == "call-1"
        assert result.state == CallState.INIT

    @pytest.mark.asyncio
    async def test_get_missing_session_raises(self, session_manager, mock_redis_client):
        mock_redis_client.get.return_value = None

        with pytest.raises(SessionError, match="not found"):
            await session_manager.get_session("nonexistent")

    @pytest.mark.asyncio
    async def test_valid_state_transition(self, session_manager, mock_redis_client):
        session = CallSession(call_id="call-1", channel_id="ch-1", state=CallState.INIT)
        mock_redis_client.get.return_value = session.model_dump_json()

        result = await session_manager.transition_state("call-1", CallState.ANSWERED)
        assert result.state == CallState.ANSWERED
        assert result.answered_at is not None

    @pytest.mark.asyncio
    async def test_invalid_state_transition_raises(self, session_manager, mock_redis_client):
        session = CallSession(call_id="call-1", channel_id="ch-1", state=CallState.INIT)
        mock_redis_client.get.return_value = session.model_dump_json()

        with pytest.raises(InvalidStateTransition):
            await session_manager.transition_state("call-1", CallState.SPEAKING)

    @pytest.mark.asyncio
    async def test_transition_to_terminated_sets_timestamp(
        self, session_manager, mock_redis_client
    ):
        session = CallSession(
            call_id="call-1", channel_id="ch-1", state=CallState.LISTENING
        )
        mock_redis_client.get.return_value = session.model_dump_json()

        result = await session_manager.transition_state("call-1", CallState.TERMINATED)
        assert result.state == CallState.TERMINATED
        assert result.terminated_at is not None

    @pytest.mark.asyncio
    async def test_transition_to_interrupted_sets_flag(
        self, session_manager, mock_redis_client
    ):
        session = CallSession(
            call_id="call-1", channel_id="ch-1", state=CallState.SPEAKING
        )
        mock_redis_client.get.return_value = session.model_dump_json()

        result = await session_manager.transition_state("call-1", CallState.INTERRUPTED)
        assert result.interrupt_flag is True

    @pytest.mark.asyncio
    async def test_delete_session(self, session_manager, mock_redis_client):
        await session_manager.delete_session("call-1")
        mock_redis_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check(self, session_manager, mock_redis_client):
        mock_redis_client.ping.return_value = True
        result = await session_manager.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, session_manager, mock_redis_client):
        mock_redis_client.ping.side_effect = ConnectionError("Redis down")
        result = await session_manager.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_lock_failure_raises(self, session_manager, mock_redis_client):
        lock_mock = AsyncMock()
        lock_mock.acquire.return_value = False
        mock_redis_client.lock.return_value = lock_mock

        with pytest.raises(SessionError, match="Failed to acquire lock"):
            await session_manager.transition_state("call-1", CallState.ANSWERED)


class TestFullCallLifecycle:
    """Integration-style test for full call state lifecycle."""

    @pytest.mark.asyncio
    async def test_happy_path_lifecycle(self, session_manager, mock_redis_client):
        """Test the happy path: INIT -> ANSWERED -> LISTENING -> THINKING -> SPEAKING -> LISTENING -> TERMINATED."""
        states = [
            CallState.INIT,
            CallState.ANSWERED,
            CallState.LISTENING,
            CallState.THINKING,
            CallState.SPEAKING,
            CallState.LISTENING,
            CallState.TERMINATED,
        ]

        # Create initial session
        mock_redis_client.exists.return_value = False
        session = await session_manager.create_session("lifecycle-1", "ch-1")

        current_state = CallState.INIT
        for target_state in states[1:]:
            session = CallSession(
                call_id="lifecycle-1",
                channel_id="ch-1",
                state=current_state,
            )
            mock_redis_client.get.return_value = session.model_dump_json()

            result = await session_manager.transition_state("lifecycle-1", target_state)
            assert result.state == target_state
            current_state = target_state

    @pytest.mark.asyncio
    async def test_interrupt_lifecycle(self, session_manager, mock_redis_client):
        """Test barge-in: SPEAKING -> INTERRUPTED -> LISTENING."""
        session = CallSession(
            call_id="int-1", channel_id="ch-1", state=CallState.SPEAKING
        )
        mock_redis_client.get.return_value = session.model_dump_json()

        result = await session_manager.transition_state("int-1", CallState.INTERRUPTED)
        assert result.state == CallState.INTERRUPTED

        mock_redis_client.get.return_value = result.model_dump_json()
        result = await session_manager.transition_state("int-1", CallState.LISTENING)
        assert result.state == CallState.LISTENING
        assert result.interrupt_flag is False
