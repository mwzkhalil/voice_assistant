"""
LLM service with streaming token generation.

Uses OpenAI AsyncClient pointed at Ollama's OpenAI-compatible endpoint.
Includes circuit breaker, exponential backoff retry, timeout enforcement,
and input sanitization.
"""

from __future__ import annotations

import asyncio
import re
import time
from enum import IntEnum
from typing import AsyncGenerator

from openai import AsyncOpenAI

from core.config import OllamaSettings
from core.exceptions import CircuitBreakerOpen, LLMError
from core.logger import get_logger
from core.metrics import MetricsCollector

logger = get_logger(__name__)


class CircuitState(IntEnum):
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading LLM failures.

    States:
        CLOSED:    Normal operation, requests pass through.
        OPEN:      Failures exceeded threshold, requests rejected immediately.
        HALF_OPEN: Recovery probe, one request allowed to test service.
    """

    def __init__(self, threshold: int, recovery_timeout: float) -> None:
        self._threshold = threshold
        self._recovery_timeout = recovery_timeout
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def check(self) -> None:
        """Check if request is allowed. Raises CircuitBreakerOpen if not."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return

            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    MetricsCollector.set_circuit_breaker_state(CircuitState.HALF_OPEN)
                    logger.info("circuit_breaker_half_open", elapsed=elapsed)
                    return
                raise CircuitBreakerOpen(
                    "llm",
                    recovery_after=self._recovery_timeout - elapsed,
                )

            # HALF_OPEN: allow one request through
            return

    async def record_success(self) -> None:
        """Record a successful request."""
        async with self._lock:
            self._failure_count = 0
            if self._state != CircuitState.CLOSED:
                self._state = CircuitState.CLOSED
                MetricsCollector.set_circuit_breaker_state(CircuitState.CLOSED)
                logger.info("circuit_breaker_closed")

    async def record_failure(self) -> None:
        """Record a failed request."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                MetricsCollector.set_circuit_breaker_state(CircuitState.OPEN)
                logger.warning("circuit_breaker_reopened", failures=self._failure_count)
            elif self._failure_count >= self._threshold:
                self._state = CircuitState.OPEN
                MetricsCollector.set_circuit_breaker_state(CircuitState.OPEN)
                logger.warning(
                    "circuit_breaker_opened",
                    failures=self._failure_count,
                    threshold=self._threshold,
                )


# Regex to strip control characters (keep Urdu, Arabic, whitespace, punctuation)
_CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def sanitize_input(text: str) -> str:
    """Remove control characters from user input while preserving Urdu text."""
    return _CONTROL_CHAR_RE.sub('', text).strip()


class LLMService:
    """
    Enterprise LLM service with streaming, circuit breaker, and retry.

    Streams tokens from Ollama's OpenAI-compatible API endpoint.
    Always injects Urdu system prompt. Enforces token limits and timeouts.
    """

    def __init__(self, settings: OllamaSettings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.request_timeout,
        )
        self._circuit_breaker = CircuitBreaker(
            threshold=settings.circuit_breaker_threshold,
            recovery_timeout=settings.circuit_breaker_recovery_timeout,
        )
        logger.info(
            "llm_service_initialized",
            model=settings.model_name,
            base_url=settings.base_url,
            max_tokens=settings.max_tokens,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
        logger.info("llm_client_closed")

    async def generate_stream(
        self,
        transcript: str,
        session_history: list[dict[str, str]],
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream LLM tokens for the given transcript with conversation history.

        Args:
            transcript: Current user utterance (ASR output).
            session_history: List of prior message dicts [{"role": ..., "content": ...}].
            cancel_event: Optional event to signal cancellation.

        Yields:
            Token strings as they are generated.

        Raises:
            LLMError: On failure after retries exhausted.
            CircuitBreakerOpen: If circuit breaker is open.
        """
        # Circuit breaker check
        await self._circuit_breaker.check()

        # Sanitize input
        clean_transcript = sanitize_input(transcript)
        if not clean_transcript:
            logger.warning("llm_empty_transcript_after_sanitization")
            return

        # Build messages
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._settings.system_prompt},
        ]
        # Add session history (bounded)
        for msg in session_history[-20:]:
            messages.append(msg)
        messages.append({"role": "user", "content": clean_transcript})

        # Retry loop with exponential backoff
        last_error: Exception | None = None
        for attempt in range(self._settings.retry_max_attempts):
            if cancel_event and cancel_event.is_set():
                return

            try:
                async for token in self._stream_once(messages, cancel_event):
                    yield token
                await self._circuit_breaker.record_success()
                return
            except CircuitBreakerOpen:
                raise
            except Exception as exc:
                last_error = exc
                await self._circuit_breaker.record_failure()
                MetricsCollector.increment_llm_error(type(exc).__name__)

                if attempt < self._settings.retry_max_attempts - 1:
                    delay = self._settings.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        "llm_retry",
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)

        raise LLMError(
            "LLM generation failed after retries",
            attempts=self._settings.retry_max_attempts,
            last_error=str(last_error),
        )

    async def _stream_once(
        self,
        messages: list[dict[str, str]],
        cancel_event: asyncio.Event | None,
    ) -> AsyncGenerator[str, None]:
        """Single attempt to stream tokens from the LLM."""
        start = time.monotonic()
        first_token = True
        token_count = 0

        try:
            stream = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._settings.model_name,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=self._settings.max_tokens,
                    temperature=self._settings.temperature,
                    stream=True,
                ),
                timeout=self._settings.request_timeout,
            )

            async for chunk in stream:
                if cancel_event and cancel_event.is_set():
                    logger.info("llm_stream_cancelled")
                    return

                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    if first_token:
                        ttft = time.monotonic() - start
                        MetricsCollector.record_llm_ttft(ttft)
                        first_token = False

                    token_count += 1
                    yield delta.content

                    if token_count >= self._settings.max_tokens:
                        logger.info("llm_max_tokens_reached", count=token_count)
                        break

            total_latency = time.monotonic() - start
            MetricsCollector.record_llm_total(total_latency)
            MetricsCollector.increment_call(str(token_count))

            logger.info(
                "llm_stream_complete",
                tokens=token_count,
                total_latency=total_latency,
            )

        except asyncio.TimeoutError:
            MetricsCollector.increment_timeout("llm")
            raise LLMError(
                "LLM request timeout",
                timeout=self._settings.request_timeout,
            )
