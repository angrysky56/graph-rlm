"""Exception handler utilities for consistent error handling.

Provides standardized exception handling patterns with logging and context preservation
and FastAPI exception handlers for HTTP response mapping.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from fastapi import Request
from fastapi.responses import JSONResponse

from ..logging import get_correlation_id
from ..logging import get_logger as get_structlog_logger
from .base import BaseGraphRLMError
from .codes import ErrorCode
from .types import (
    CoreError,
    ExternalServiceError,
    ValidationError,
)

T = TypeVar("T", bound=BaseGraphRLMError)


class ExceptionHandler:
    """Standardized exception handler with logging and context."""

    def __init__(self, *, log_level: str = "error"):
        self.logger = get_structlog_logger(__name__)
        self.log_level = log_level

    def handle(
        self,
        exception: BaseException,
        *,
        error_code: ErrorCode,
        message: str | None = None,
        reraise_as: type[T] | None = None,
        context: dict[str, Any] | None = None,
        log_message: str | None = None,
    ) -> T | None:
        """Handle an exception consistently.

        Args:
            exception: The caught exception
            error_code: ErrorCode for categorization
            message: Override message for the new exception
            reraise_as: Specific exception type to raise
            context: Additional context to include
            log_message: Custom log message

        Returns:
            The exception (possibly wrapped) if not reraised
        """
        correlation_id = get_correlation_id()

        handler_context: dict[str, Any] = {
            "error_code": error_code.value,
            "correlation_id": correlation_id,
            "exception_type": type(exception).__name__,
        }
        if context:
            handler_context.update(context)

        log_msg = log_message or f"{type(exception).__name__}: {str(exception)}"
        log_method = getattr(self.logger, self.log_level)
        log_method(log_msg, exc_info=True, **handler_context)

        if reraise_as:
            raise reraise_as(
                message=message or str(exception),
                error_code=error_code,
                correlation_id=correlation_id,
                cause=exception,
                **(context or {}),
            ) from exception

        return None


def safe_call(
    func: Callable[..., T],
    *,
    error_code: ErrorCode,
    fallback: T | None = None,
    reraise: bool = False,
    context: dict[str, Any] | None = None,
) -> T | None:
    """Execute a function with standardized exception handling.

    Args:
        func: Function to execute
        error_code: ErrorCode for potential exceptions
        fallback: Value to return on exception
        reraise: Whether to reraise as Graph-RLM exception
        context: Additional context for exception

    Returns:
        Function result or fallback
    """
    handler = ExceptionHandler()
    correlation_id = get_correlation_id()

    handler_context = dict(context or {})
    handler_context["correlation_id"] = correlation_id
    handler_context["function"] = func.__name__

    try:
        return func()
    except BaseGraphRLMError:
        raise
    except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
        handler.handle(
            e,
            error_code=error_code,
            context=handler_context,
            reraise_as=CoreError if reraise else None,
        )
        return fallback


def wrap_operation(
    operation_name: str,
    *,
    error_code: ErrorCode,
    category: str = "operation",
) -> Callable:
    """Decorator to wrap a function with exception handling.

    Args:
        operation_name: Name of the operation for context
        error_code: ErrorCode to use
        category: Category for context

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            handler = ExceptionHandler()
            correlation_id = get_correlation_id()

            context = {
                "operation": operation_name,
                "category": category,
                "correlation_id": correlation_id,
            }

            try:
                return func(*args, **kwargs)
            except BaseGraphRLMError:
                raise
            except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
                handler.handle(
                    e,
                    error_code=error_code,
                    context=context,
                    reraise_as=CoreError,
                )
                return None

        return wrapper

    return decorator


# =============================================================================
# FastAPI Exception Handlers
# =============================================================================


async def graphrlm_exception_handler(
    _: Request, exc: BaseGraphRLMError
) -> JSONResponse:
    """Handle all GraphRLM exceptions with proper HTTP status codes.

    This is the generic handler for BaseGraphRLMError and its subclasses.
    More specific handlers should be registered for ValidationError, etc.
    to ensure proper status code mapping.
    """
    # Get http_status_code if the exception has it, otherwise default to 500
    status_code = getattr(exc, "http_status_code", 500)

    return JSONResponse(status_code=status_code, content=exc.to_dict())


async def validation_exception_handler(
    _: Request, exc: ValidationError
) -> JSONResponse:
    """Handle validation errors with 422 status code."""
    # Build response content including validation details from context
    response_content = exc.to_dict()

    return JSONResponse(status_code=422, content=response_content)


async def circuit_open_exception_handler(
    _: Request, exc: BaseGraphRLMError
) -> JSONResponse:
    """Handle circuit breaker open errors with 503 status code."""
    # Lazy import to avoid circular dependency
    from ..circuit import CircuitOpenError as CircuitOpenErrorClass

    # Ensure this handler is only called for CircuitOpenError
    if not isinstance(exc, CircuitOpenErrorClass):
        raise TypeError(f"Expected CircuitOpenError, got {type(exc).__name__}")

    # CircuitOpenError may have circuit_name in context
    response_content = exc.to_dict()

    return JSONResponse(status_code=503, content=response_content)


async def external_service_exception_handler(
    _: Request, exc: ExternalServiceError
) -> JSONResponse:
    """Handle external service errors with 503 status code."""
    # ExternalServiceError may have service/endpoint info in context
    response_content = exc.to_dict()

    return JSONResponse(status_code=503, content=response_content)
