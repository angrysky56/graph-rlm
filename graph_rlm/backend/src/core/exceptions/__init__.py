"""Graph-RLM Exception Hierarchy.

Exports all exception classes and error codes for public use.

Example:
    from graph_rlm.backend.src.core.exceptions import (
        BaseGraphRLMError,
        GraphError,
        SkillExecutionError,
        ErrorCode,
        ValidationError,
    )

    raise GraphError(
        message="Node not found",
        error_code=ErrorCode.GRAPH_NODE_NOT_FOUND,
        correlation_id="req-123",
        node_id="user:123",
    )
"""

from .base import BaseGraphRLMError, GraphRLMExceptionContext
from .codes import ErrorCode, ErrorCodeCategory
from .types import (
    CoreError,
    ExternalServiceError,
    GraphError,
    SkillExecutionError,
    ValidationError,
)

__all__ = [
    # Base classes
    "BaseGraphRLMError",
    "GraphRLMExceptionContext",
    # Error codes
    "ErrorCode",
    "ErrorCodeCategory",
    # Specific types
    "CoreError",
    "GraphError",
    "SkillExecutionError",
    "ExternalServiceError",
    "ValidationError",
]
