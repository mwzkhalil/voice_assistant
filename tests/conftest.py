"""
Pytest fixtures for the voice assistant test suite.

Provides mock services, test configurations, and utilities
for isolated unit and integration testing.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from core.config import (
    AppSettings,
    ARISettings,
    ASRSettings,
    OllamaSettings,
    RedisSettings,
    RTPSettings,
    ServerSettings,
    TTSSettings,
    VADSettings,
    reset_settings,
)
from core.logger import configure_logging, LogLevel


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset all module-level singletons between tests."""
    from core.dependency_injection import reset_container
    from core.lifecycle import reset_lifecycle

    reset_settings()
    reset_container()
    reset_lifecycle()
    yield
    reset_settings()
    reset_container()
    reset_lifecycle()


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use default event loop policy for tests."""
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
def test_settings() -> AppSettings:
    """Create test-specific settings with safe defaults."""
    return AppSettings(
        ari=ARISettings(
            host="localhost",
            port=8088,
            username="test",
            password="test",
            app_name="test-app",
        ),
        redis=RedisSettings(
            url="redis://localhost:6379/15",  # Use DB 15 for tests
            pool_max_size=2,
            session_ttl=60,
        ),
        ollama=OllamaSettings(
            base_url="http://localhost:11434/v1",
            model_name="test-model",
            max_tokens=50,
            request_timeout=5.0,
        ),
        asr=ASRSettings(
            model_name="test-asr",
            device="cpu",
            chunk_duration_ms=1000,
        ),
        tts=TTSSettings(
            model_name="test-tts",
            device="cpu",
        ),
        vad=VADSettings(
            aggressiveness=2,
            silence_threshold_ms=500,
        ),
        rtp=RTPSettings(
            port_range_start=30000,
            port_range_end=30100,
        ),
        server=ServerSettings(
            port=9999,
            log_level=LogLevel.DEBUG,
            debug=True,
            max_concurrent_calls=10,
        ),
    )


@pytest.fixture
def mock_redis():
    """Create a mock Redis client.

    FIX: redis.lock() is a synchronous call that *returns* a lock object.
    Only lock.acquire() / lock.release() are async.
    Using AsyncMock for .lock itself caused it to return a coroutine instead
    of a lock object, so calling .acquire() on the coroutine raised:
        AttributeError: 'coroutine' object has no attribute 'acquire'
    """
    mock = AsyncMock()
    mock.ping.return_value = True
    mock.get.return_value = None
    mock.set.return_value = True
    mock.delete.return_value = 1
    mock.exists.return_value = False
    mock.scan.return_value = (0, [])

    # Build a proper lock mock:
    #   - lock.acquire() / release() are async  → AsyncMock
    #   - client.lock() itself is sync          → MagicMock(return_value=...)
    lock_mock = MagicMock()
    lock_mock.acquire = AsyncMock(return_value=True)
    lock_mock.release = AsyncMock(return_value=None)
    # Support use as an async context manager: `async with client.lock(...)`
    lock_mock.__aenter__ = AsyncMock(return_value=lock_mock)
    lock_mock.__aexit__ = AsyncMock(return_value=False)

    # MagicMock (not AsyncMock) so calling it returns lock_mock synchronously
    mock.lock = MagicMock(return_value=lock_mock)

    return mock


@pytest.fixture
def mock_redis_pool(mock_redis):
    """Create a mock Redis connection pool."""
    pool = MagicMock()
    pool.aclose = AsyncMock()
    return pool


@pytest.fixture
def mock_ari_client():
    """Create a mock ARI client."""
    mock = AsyncMock()
    mock.is_connected = True
    mock.connect = AsyncMock()
    mock.disconnect = AsyncMock()
    mock.answer_channel = AsyncMock(return_value={})
    mock.hangup_channel = AsyncMock(return_value={})
    mock.create_bridge = AsyncMock(return_value={"id": "test-bridge-id"})
    mock.add_channel_to_bridge = AsyncMock(return_value={})
    mock.create_external_media = AsyncMock(return_value={
        "channel": {"id": "ext-media-channel-id"},
    })
    mock.destroy_bridge = AsyncMock(return_value={})
    mock.get_events = AsyncMock()
    return mock


@pytest.fixture
def mock_asr_service():
    """Create a mock ASR service."""
    mock = AsyncMock()
    mock.is_loaded = True
    mock.load_model = AsyncMock()
    mock.unload_model = AsyncMock()
    mock.transcribe_chunk = AsyncMock(return_value="ٹیسٹ ٹرانسکرپشن")
    return mock


@pytest.fixture
def mock_tts_service():
    """Create a mock TTS service."""
    mock = AsyncMock()
    mock.is_loaded = True
    mock.load_model = AsyncMock()
    mock.unload_model = AsyncMock()
    mock.synthesize_text = AsyncMock(return_value=b"\x00" * 1600)
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    mock = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_vad_service():
    """Create a mock VAD service.

    FIX: importing from services.vad_service at fixture-definition time caused
    ModuleNotFoundError when webrtcvad is not installed, because the import
    runs even for tests that never use this fixture.

    Instead we mock out the entire vad_service module so the real webrtcvad
    C-extension is never imported during the test run.
    """
    # Stub the two named types the fixture previously imported
    VADEvent = MagicMock()
    VADEvent.SILENCE = "SILENCE"
    VADEvent.SPEECH_START = "SPEECH_START"
    VADEvent.SPEECH_END = "SPEECH_END"

    VADResult = MagicMock(return_value=MagicMock(
        event=VADEvent.SILENCE,
        is_speech=False,
        speech_duration_ms=0,
        silence_duration_ms=0,
        should_trigger_asr=False,
        should_finalize=False,
    ))

    tracker_mock = MagicMock()
    tracker_mock.is_speaking = False
    tracker_mock.reset = MagicMock()
    tracker_mock.process_frame = MagicMock(return_value=VADResult())

    mock = MagicMock()
    mock.create_tracker.return_value = tracker_mock
    mock.frame_size_bytes = 960   # 30 ms at 16 kHz
    mock.frame_duration_ms = 30
    return mock
