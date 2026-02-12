"""Base exception class for Graph-RLM with context preservation.

Provides a foundation for all Graph-RLM exceptions with:
- Error code support
- Correlation ID for request tracing
- Timestamp tracking
- Cause chaining (exception wrapping)
- Context metadata storage
- Structured dict serialization
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from .codes import ErrorCode


class GraphRLMExceptionContext:
    """Container for exception context metadata.

    Provides a flexible dict-like structure for storing
    operation-specific context that aids debugging.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._data: dict[str, Any] = kwargs

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"GraphRLMExceptionContext({self._data!r})"

    def to_dict(self) -> dict[str, Any]:
        """Convert context to serializable dict."""
        return self._data.copy()

    def merge(self, other: "GraphRLMExceptionContext") -> "GraphRLMExceptionContext":
        """Merge another context into this one."""
        merged = GraphRLMExceptionContext(**self._data)
        merged._data.update(other._data)
        return merged


class BaseGraphRLMError(Exception):
    """Base exception for all Graph-RLM errors.

    Provides:
    - Error code categorization
    - Correlation ID for request tracing
    - UTC timestamp
    - Exception chaining (cause preservation)
    - Context metadata
    - Structured serialization
    """

    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        *,
        correlation_id: Optional[str] = None,
        cause: Optional[BaseException] = None,
        **context: Any,
    ) -> None:
        """Initialize the exception with full context.

        Args:
            message: Human-readable error message
            error_code: ErrorCode enum value
            correlation_id: Request correlation ID for tracing
            cause: Original exception for chaining
            **context: Additional context metadata
        """
        super().__init__(message, error_code, correlation_id, cause)

        self._message = message
        self._error_code = error_code
        self._correlation_id = correlation_id
        self._timestamp = datetime.now(timezone.utc)
        self._context = (
            GraphRLMExceptionContext(**context)
            if context
            else GraphRLMExceptionContext()
        )

        if cause is not None:
            self.__cause__ = cause
            self.add_context("cause_error_type", type(cause).__name__)
            self.add_context("cause_message", str(cause))

    @property
    def message(self) -> str:
        """Get the error message."""
        return self._message

    @property
    def error_code(self) -> ErrorCode:
        """Get the error code."""
        return self._error_code

    @property
    def correlation_id(self) -> Optional[str]:
        """Get the correlation ID."""
        return self._correlation_id

    @property
    def timestamp(self) -> datetime:
        """Get the UTC timestamp."""
        return self._timestamp

    @property
    def context(self) -> GraphRLMExceptionContext:
        """Get the context metadata."""
        return self._context

    def with_correlation_id(self, correlation_id: str) -> "BaseGraphRLMError":
        """Create a new exception with a correlation ID.

        Useful for adding tracing to re-raised exceptions.
        """
        new_exc = self.__class__(
            message=self._message,
            error_code=self._error_code,
            correlation_id=correlation_id,
            cause=self.__cause__,
        )
        new_exc._context = self._context.merge(new_exc._context)
        return new_exc

    def with_context(self, **context: Any) -> "BaseGraphRLMError":
        """Create a new exception with additional context.

        Useful for enriching exceptions with operation-specific data.
        """
        new_exc = self.__class__(
            message=self._message,
            error_code=self._error_code,
            correlation_id=self._correlation_id,
            cause=self.__cause__,
            **self._context.to_dict(),
        )
        new_exc._context = self._context.merge(GraphRLMExceptionContext(**context))
        return new_exc

    def add_context(self, key: str, value: Any) -> None:
        """Add context to this exception instance."""
        self._context[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert to a serializable dictionary.

        Suitable for JSON serialization in API responses.
        """
        return {
            "error_type": self.__class__.__name__,
            "error_code": self._error_code.value,
            "message": self._message,
            "correlation_id": self._correlation_id,
            "timestamp": self._timestamp.isoformat(),
            "context": self._context.to_dict(),
            "cause": {
                "type": type(self.__cause__).__name__ if self.__cause__ else None,
                "message": str(self.__cause__) if self.__cause__ else None,
            }
            if self.__cause__
            else None,
        }

    def to_json(self, **kwargs) -> str:
        """Convert exception to JSON string.

        Args:
            kwargs: Additional arguments for json.dumps

        Returns:
            JSON string representation
        """
        import json

        return json.dumps(self.to_dict(), **kwargs)

    def format_traceback(self) -> str:
        """Format exception traceback as string."""
        import traceback

        return (
            "".join(
                traceback.format_exception(type(self), self, self.__traceback__)
            ).strip()
            if self.__traceback__
            else ""
        )

    def __str__(self) -> str:
        return f"[{self._error_code.value}] {self._message}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self._message!r}, "
            f"error_code={self._error_code.value!r}, "
            f"correlation_id={self._correlation_id!r}"
            f")"
        )

    def __reduce__(self):
        """Support for pickling."""
        return (
            self.__class__,
            (self._message, self._error_code),
            {
                "correlation_id": self._correlation_id,
                "context": self._context.to_dict(),
            },
        )
