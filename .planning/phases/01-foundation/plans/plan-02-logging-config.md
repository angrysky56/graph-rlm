---
wave: 2
depends_on: []
autonomous: true
files_modified:
  - graph_rlm/backend/src/core/logging.py (new)
  - graph_rlm/backend/src/main.py (update for logging setup)
---

# Phase 1: Foundation - Structured Logging Configuration

## Overview

Configure structlog for JSON-structured logging with correlation ID propagation and async compatibility. This establishes the logging infrastructure used by exception handlers.

## Requirements Addressed

- LOG-01: structlog configuration for JSON-structured logging
- LOG-02: Exception context enrichment for all logs

## Implementation Order

1. Create logging configuration module
2. Set up structlog processors (timestamp, log level, context enrichment)
3. Configure JSON renderer for production, console for development
4. Set up correlation ID context variable
5. Integrate with main application startup

## Tasks

### 2.1 Create logging configuration (logging.py)

Create graph_rlm/backend/src/core/logging.py:

```python
"""Structured logging configuration for Graph-RLM."""

import logging as stdlib_logging
import os
from typing import Any

import structlog
from structlog import processors as structlog_processors
from structlog.contextvars import merge_contextvars, clear_contextvars, bind_contextvars
from structlog.stdlib import add_log_level, ProcessorFormatter

def get_correlation_id() -> str | None:
    """Get current correlation ID from context variable."""
    from contextvars import ContextVar
    correlation_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)
    return correlation_var.get()

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
    
    # Pre-processors applied before logging
    processors: list[structlog.types.Processor] = [
        merge_contextvars,
        add_log_level,
        structlog_processors.TimeStamper(fmt="iso", utc=True),
        structlog_processors.CallsiteParameterGetter(
            structlog_processors.CallsiteParameter.FILENAME,
            structlog_processors.CallsiteParameter.LINENO,
            structlog_processors.CallsiteParameter.FUNC_NAME,
        ),
    ]
    
    # Add exception context enrichment
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
        return event_dict
    
    processors.append(enrich_with_exception)
    
    # Post-processors for rendering
    if env == "production":
        # JSON output for production (log aggregation)
        post_processors = [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ]
        renderer = structlog.processors.JSONRenderer()
    else:
        # Colored console output for development
        post_processors = [
            structlog.processors.LogfmtRenderer(),
        ]
        renderer = structlog.dev.ConsoleRenderer(
            colors=True, 
            timestamp_format="%H:%M:%S"
        )
    
    # Configure structlog
    structlog.configure(
        processors=processors + post_processors,
        wrapper_class=structlog.stdlib.ProcessorWrapper,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    root_logger = stdlib_logging.getLogger()
    handler = stdlib_logging.StreamHandler()
    handler.setFormatter(ProcessorFormatter(
        processors=processors,
        foreign_pre_chain=processors,
    ))
    root_logger.addHandler(handler)
    root_logger.setLevel(stdlib_logging.DEBUG if env != "production" else stdlib_logging.INFO)

def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance.
    
    Args:
        name: Optional logger name (uses module name if not provided)
    
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)

# Context variable for correlation ID propagation
CORRELATION_ID_CTX: structlog.contextvars.ContextVar[str | None] = structlog.contextvars.ContextVar(
    "correlation_id", default=None
)

def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID in context variable."""
    CORRELATION_ID_CTX.set(correlation_id)
    bind_contextvars(correlation_id=correlation_id)

def get_correlation_id() -> str | None:
    """Get correlation ID from context variable."""
    return CORRELATION_ID_CTX.get()
