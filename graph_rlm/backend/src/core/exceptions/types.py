"""Specific exception types for Graph-RLM.

Provides targeted exception classes for different error categories:
- CoreError: System-level errors
- GraphError: Graph operation errors
- SkillExecutionError: Skill execution errors
- ExternalServiceError: External service errors
- ValidationError: Input validation errors
"""

from __future__ import annotations

from typing import Any, Optional

from .base import BaseGraphRLMError
from .codes import ErrorCode, ErrorCodeCategory


class CoreError(BaseGraphRLMError):
    """System-level errors (CORE_* error codes)."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        *,
        correlation_id: Optional[str] = None,
        cause: Optional[BaseException] = None,
        **context: Any,
    ) -> None:
        if not error_code.category == ErrorCodeCategory.CORE:
            error_code = ErrorCode.CORE_INTERNAL_ERROR

        super().__init__(
            message=message,
            error_code=error_code,
            correlation_id=correlation_id,
            cause=cause,
            **context,
        )

    def with_operation(self, operation: str) -> "CoreError":
        """Add operation context."""
        return self.with_context(operation=operation)


class GraphError(BaseGraphRLMError):
    """Graph operation errors (GRAPH_* error codes)."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        *,
        correlation_id: Optional[str] = None,
        cause: Optional[BaseException] = None,
        **context: Any,
    ) -> None:
        if not error_code.category == ErrorCodeCategory.GRAPH:
            error_code = ErrorCode.GRAPH_OPERATION_FAILED

        super().__init__(
            message=message,
            error_code=error_code,
            correlation_id=correlation_id,
            cause=cause,
            **context,
        )

    def with_graph_operation(self, operation: str) -> "GraphError":
        """Add graph operation context."""
        return self.with_context(graph_operation=operation)

    def with_node_id(self, node_id: str) -> "GraphError":
        """Add node ID context."""
        return self.with_context(node_id=node_id)

    def with_edge(self, source: str, target: str) -> "GraphError":
        """Add edge context."""
        return self.with_context(source=source, target=target)


class SkillExecutionError(BaseGraphRLMError):
    """Skill execution errors (SKILL_* error codes)."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        *,
        correlation_id: Optional[str] = None,
        cause: Optional[BaseException] = None,
        **context: Any,
    ) -> None:
        if not error_code.category == ErrorCodeCategory.SKILL:
            error_code = ErrorCode.SKILL_EXECUTION_FAILED

        super().__init__(
            message=message,
            error_code=error_code,
            correlation_id=correlation_id,
            cause=cause,
            **context,
        )

    def with_skill_name(self, skill_name: str) -> "SkillExecutionError":
        """Add skill name context."""
        return self.with_context(skill_name=skill_name)

    def with_skill_input(self, skill_input: Any) -> "SkillExecutionError":
        """Add skill input context (sanitized)."""
        return self.with_context(skill_input=str(skill_input)[:1000])

    def with_skill_output(self, skill_output: Any) -> "SkillExecutionError":
        """Add skill output context (sanitized)."""
        return self.with_context(skill_output=str(skill_output)[:1000])


class ExternalServiceError(BaseGraphRLMError):
    """External service errors (EXTERNAL_* error codes)."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        *,
        correlation_id: Optional[str] = None,
        cause: Optional[BaseException] = None,
        **context: Any,
    ) -> None:
        if not error_code.category == ErrorCodeCategory.EXTERNAL:
            error_code = ErrorCode.EXTERNAL_SERVICE_ERROR

        super().__init__(
            message=message,
            error_code=error_code,
            correlation_id=correlation_id,
            cause=cause,
            **context,
        )

    def with_service_name(self, service_name: str) -> "ExternalServiceError":
        """Add service name context."""
        return self.with_context(service=service_name)

    def with_endpoint(self, endpoint: str) -> "ExternalServiceError":
        """Add endpoint context."""
        return self.with_context(endpoint=endpoint)

    def with_request(self, method: str, url: str) -> "ExternalServiceError":
        """Add request context."""
        return self.with_context(method=method, url=url)

    def with_response_status(self, status_code: int) -> "ExternalServiceError":
        """Add response status context."""
        return self.with_context(status_code=status_code)

    @property
    def http_status_code(self) -> int:
        """Return 503 for external service errors."""
        return 503


class ValidationError(BaseGraphRLMError):
    """Input validation errors (VALIDATION_* error codes)."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        *,
        correlation_id: Optional[str] = None,
        cause: Optional[BaseException] = None,
        **context: Any,
    ) -> None:
        if not error_code.category == ErrorCodeCategory.VALIDATION:
            error_code = ErrorCode.VALIDATION_FIELD_INVALID

        super().__init__(
            message=message,
            error_code=error_code,
            correlation_id=correlation_id,
            cause=cause,
            **context,
        )

    def with_field_errors(self, field_errors: dict[str, str]) -> "ValidationError":
        """Add field-specific errors."""
        return self.with_context(field_errors=field_errors)

    def with_field(self, field: str, value: Any) -> "ValidationError":
        """Add field and value context."""
        return self.with_context(field=field, field_value=str(value)[:500])

    def with_schema(self, schema_name: str) -> "ValidationError":
        """Add schema context."""
        return self.with_context(schema=schema_name)

    def with_constraint(self, constraint: str) -> "ValidationError":
        """Add constraint context."""
        return self.with_context(constraint=constraint)

    @property
    def http_status_code(self) -> int:
        """Return 422 for validation errors."""
        return 422
