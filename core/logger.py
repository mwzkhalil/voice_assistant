"""
Enterprise structured logging via structlog.

JSON-formatted output with correlation IDs, log levels, and caller information.
No print statements allowed anywhere in the codebase.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog

from core.config import LogLevel, get_settings

# Context variable for per-request / per-call correlation
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="none")
call_id_var: ContextVar[str] = ContextVar("call_id", default="none")


def _add_correlation_id(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Inject correlation_id from contextvars into every log entry."""
    event_dict["correlation_id"] = correlation_id_var.get()
    return event_dict


def _add_call_id(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Inject call_id from contextvars into every log entry."""
    cid = call_id_var.get()
    if cid != "none":
        event_dict["call_id"] = cid
    return event_dict


def _add_service_context(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add static service identification."""
    event_dict["service"] = "voice-assistant"
    return event_dict


def configure_logging(level: LogLevel | None = None) -> None:
    """
    Configure structlog for JSON-formatted production logging.

    Must be called once at application startup.
    """
    if level is None:
        level = get_settings().server.log_level

    log_level = getattr(logging, level.value, logging.INFO)

    # Configure stdlib logging to route through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Silence noisy third-party loggers
    for noisy in ("websockets", "urllib3", "httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            _add_service_context,
            _add_correlation_id,
            _add_call_id,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.MODULE,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ],
            ),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Obtain a named, bound structlog logger.

    Args:
        name: Logger name, typically the module name.

    Returns:
        A configured bound logger instance.
    """
    return structlog.get_logger(name)


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current async context."""
    correlation_id_var.set(cid)


def set_call_id(cid: str) -> None:
    """Set the call ID for the current async context."""
    call_id_var.set(cid)
