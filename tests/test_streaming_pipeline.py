"""
Tests for the streaming pipeline orchestrator.

Covers pipeline start/stop, interrupt handling, and queue wiring.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from services.streaming_pipeline import StreamingPipeline
from services.session_manager import CallSession, CallState


@pytest.fixture
def mock_transport():
    """Create a mock AudioTransport."""
    mock = AsyncMock()
    mock.read_audio = AsyncMock(return_value=None)
    mock.write_audio = AsyncMock()
    mock.stop_output = MagicMock()
    mock.resume_output = MagicMock()
    mock.start = AsyncMock()
    mock.stop = AsyncMock()
    mock.set_transcript_callback = MagicMock()
    return mock


@pytest.fixture
def mock_session_manager():
    """Create a mock session manager."""
    session = CallSession(
        call_id="test-call",
        channel_id="test-ch",
        state=CallState.LISTENING,
    )
    mock = AsyncMock()
    mock.get_session = AsyncMock(return_value=session)
    mock.update_session = AsyncMock()
    mock.transition_state = AsyncMock(return_value=session)
    return mock


@pytest.fixture
def pipeline(
    mock_transport,
    mock_asr_service,
    mock_llm_service,
    mock_tts_service,
    mock_vad_service,
    mock_session_manager,
):
    """Create a StreamingPipeline with all mocked dependencies."""
    return StreamingPipeline(
        call_id="test-call",
        transport=mock_transport,
        asr_service=mock_asr_service,
        llm_service=mock_llm_service,
        tts_service=mock_tts_service,
        vad_service=mock_vad_service,
        session_manager=mock_session_manager,
    )


class TestPipelineLifecycle:
    """Tests for pipeline start/stop."""

    @pytest.mark.asyncio
    async def test_start_creates_tasks(self, pipeline):
        await pipeline.start()
        assert pipeline.is_running
        assert len(pipeline._tasks) == 5
        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_stop_sets_cancel_event(self, pipeline):
        await pipeline.start()
        await pipeline.stop()
        assert not pipeline.is_running
        assert pipeline._cancel_event.is_set()

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self, pipeline):
        await pipeline.start()
        await pipeline.start()  # Should not create duplicate tasks
        assert len(pipeline._tasks) == 5
        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start(self, pipeline):
        # Should not raise
        await pipeline.stop()
        assert not pipeline.is_running


class TestPipelineInterrupt:
    """Tests for barge-in interrupt handling."""

    @pytest.mark.asyncio
    async def test_interrupt_cancels_tts(self, pipeline, mock_session_manager):
        await pipeline.start()

        # Simulate interrupt
        await pipeline.interrupt()

        # Verify state transitions were called
        assert mock_session_manager.transition_state.call_count >= 2
        calls = mock_session_manager.transition_state.call_args_list
        states = [call.args[1] for call in calls]
        assert CallState.INTERRUPTED in states
        assert CallState.LISTENING in states

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_interrupt_stops_transport_output(
        self, pipeline, mock_transport
    ):
        await pipeline.start()
        await pipeline.interrupt()
        mock_transport.stop_output.assert_called_once()
        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_interrupt_resets_vad(
        self, pipeline, mock_vad_service
    ):
        await pipeline.start()
        tracker = mock_vad_service.create_tracker.return_value
        await pipeline.interrupt()
        tracker.reset.assert_called()
        await pipeline.stop()
