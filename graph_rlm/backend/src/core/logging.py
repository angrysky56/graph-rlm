"""Structured logging configuration for Graph-RLM.

Provides JSON-structured logging with correlation ID propagation and async compatibility.
"""

from __future__ import annotations

import logging as stdlib_logging
import os
from typing import Any

import structlog
from structlog import processors as structlog_processors
from structlog.contextvars import merge_contextvars, clear_contextvars, bind_contextvars
from structlog.stdlib import add_log_level, ProcessorFormatter

CORRELATION_ID_CTX: structlog.contextvars.ContextVar[str | None] = (
    structlog.contextvars.ContextVar("correlation_id", default=None)
)


def get_correlation_id() -> str | None:
    """Get current correlation ID from context variable."""
    return CORRELATION_ID_CTX.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID in context variable."""
    CORRELATION_ID_CTX.set(correlation_id)
    bind_contextvars(correlation_id=correlation_id)


def clear_correlation_id() -> None:
    """Clear correlation ID from context variable."""
    CORRELATION_ID_CTX.set(None)


def setup_logging(
    *,
    env: str | None = None,
    service_name: str = "graph-rlm",
) -> None:
    """Configure structlog for the application.

    Args:
        env: Environment name (defaults to GRAPH_RLM_ENV or 'development')
        service_name: Service name for log output
    """
    if env is None:
        env = os.getenv("GRAPH_RLM_ENV", "development")

    def enrich_with_exception(
        logger: Any, method_name: str, event_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Add exception context to log entries."""
        exc_info = event_dict.get("exc_info")
        if exc_info and isinstance(exc_info, BaseException):
            if hasattr(exc_info, "correlation_id"):
                event_dict["correlation_id"] = exc_info.correlation_id
            if hasattr(exc_info, "error_code"):
                event_dict["error_code"] = (
                    exc_info.error_code.value if exc_info.error_code else None
                )
            if hasattr(exc_info, "context"):
                event_dict["exception_context"] = exc_info.context.to_dict()
        return event_dict

    processors: list[structlog.types.Processor] = [
        merge_contextvars,
        add_log_level,
        structlog_processors.TimeStamper(fmt="iso", utc=True),
        structlog_processors.CallsiteParameterGetter(
            structlog_processors.CallsiteParameter.FILENAME,
            structlog_processors.CallsiteParameter.LINENO,
            structlog_processors.CallsiteParameter.FUNC_NAME,
        ),
        enrich_with_exception,
    ]

    if env == "production":
        post_processors = [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ]
    else:
        post_processors = [
            structlog.processors.LogfmtRenderer(),
        ]

    structlog.configure(
        processors=processors + post_processors,
        wrapper_class=structlog.stdlib.ProcessorWrapper,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    root_logger = stdlib_logging.getLogger()
    handler = stdlib_logging.StreamHandler()
    handler.setFormatter(
        ProcessorFormatter(
            processors=processors,
            foreign_pre_chain=processors,
        )
    )
    root_logger.addHandler(handler)
    root_logger.setLevel(
        stdlib_logging.DEBUG if env != "production" else stdlib_logging.INFO
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance.

    Args:
        name: Optional logger name (uses module name if not provided)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)
