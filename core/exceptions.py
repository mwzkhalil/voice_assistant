"""
Structured exception hierarchy for the voice assistant.

Every exception carries a context dict for structured logging.
No unhandled exceptions allowed in production.
"""

from __future__ import annotations

from typing import Any


class VoiceAssistantError(Exception):
    """Base exception for all voice assistant errors."""

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        self.context = context or {}
        self.context["error_type"] = self.__class__.__name__
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            **self.context,
        }


# --- ARI / Telephony Errors ---


class ARIConnectionError(VoiceAssistantError):
    """Failed to connect or maintain ARI WebSocket connection."""

    def __init__(self, message: str = "ARI connection failed", **ctx: Any) -> None:
        super().__init__(message, context=ctx)


class ARICommandError(VoiceAssistantError):
    """ARI REST command execution failed."""

    def __init__(self, message: str = "ARI command failed", **ctx: Any) -> None:
        super().__init__(message, context=ctx)


class RTPError(VoiceAssistantError):
    """RTP bridge or packet handling error."""

    def __init__(self, message: str = "RTP error", **ctx: Any) -> None:
        super().__init__(message, context=ctx)


# --- Session Errors ---


class SessionError(VoiceAssistantError):
    """Session creation, retrieval, or persistence error."""

    def __init__(self, message: str = "Session error", **ctx: Any) -> None:
        super().__init__(message, context=ctx)


class InvalidStateTransition(SessionError):
    """Attempted an invalid call state transition."""

    def __init__(
        self,
        current_state: str,
        target_state: str,
        call_id: str = "",
    ) -> None:
        super().__init__(
            message=f"Invalid state transition: {current_state} -> {target_state}",
            call_id=call_id,
            current_state=current_state,
            target_state=target_state,
        )


# --- Model / Service Errors ---


class ASRError(VoiceAssistantError):
    """ASR model inference or processing error."""

    def __init__(self, message: str = "ASR error", **ctx: Any) -> None:
        super().__init__(message, context=ctx)


class LLMError(VoiceAssistantError):
    """LLM service error (timeout, circuit breaker, inference failure)."""

    def __init__(self, message: str = "LLM error", **ctx: Any) -> None:
        super().__init__(message, context=ctx)


class TTSError(VoiceAssistantError):
    """TTS model inference or processing error."""

    def __init__(self, message: str = "TTS error", **ctx: Any) -> None:
        super().__init__(message, context=ctx)


# --- Resilience Errors ---


class CircuitBreakerOpen(VoiceAssistantError):
    """Circuit breaker is open; service calls are being rejected."""

    def __init__(self, service_name: str, recovery_after: float = 0.0) -> None:
        super().__init__(
            message=f"Circuit breaker open for {service_name}",
            context={
                "service_name": service_name,
                "recovery_after_seconds": recovery_after,
            },
        )


class ResourceExhausted(VoiceAssistantError):
    """System resource limit reached (GPU OOM, max calls, etc.)."""

    def __init__(self, resource: str, limit: Any = None) -> None:
        super().__init__(
            message=f"Resource exhausted: {resource}",
            context={"resource": resource, "limit": limit},
        )


# --- Pipeline Errors ---


class PipelineError(VoiceAssistantError):
    """Streaming pipeline orchestration error."""

    def __init__(self, message: str = "Pipeline error", **ctx: Any) -> None:
        super().__init__(message, context=ctx)


class PipelineTimeout(PipelineError):
    """A pipeline stage exceeded its timeout."""

    def __init__(self, stage: str, timeout: float, **ctx: Any) -> None:
        super().__init__(
            message=f"Pipeline stage '{stage}' timed out after {timeout}s",
            stage=stage,
            timeout=timeout,
            **ctx,
        )
