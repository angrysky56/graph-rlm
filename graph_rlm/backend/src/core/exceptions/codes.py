"""Error code enumeration for Graph-RLM exception hierarchy.

Provides a hierarchical enum of error codes across 5 categories:
- CORE: System-level errors
- GRAPH: Graph operation errors
- SKILL: Skill execution errors
- EXTERNAL: External service errors
- VALIDATION: Input validation errors
"""

from __future__ import annotations

from enum import Enum


class ErrorCodeCategory(str, Enum):
    """Category of error code."""

    CORE = "CORE"
    GRAPH = "GRAPH"
    SKILL = "SKILL"
    EXTERNAL = "EXTERNAL"
    VALIDATION = "VALIDATION"


class ErrorCode(str, Enum):
    """Hierarchical error codes for Graph-RLM exceptions.

    Each error code is a string value (e.g., "GRAPH_101") that can be
    parsed to extract category and numeric identifier for programmatic use.
    """

    # CORE error codes (100-199): System-level errors
    CORE_INTERNAL_ERROR = "CORE_100"
    CORE_UNEXPECTED_STATE = "CORE_101"
    CORE_CONFIGURATION_ERROR = "CORE_102"
    CORE_DEPENDENCY_MISSING = "CORE_103"
    CORE_RESOURCE_EXHAUSTED = "CORE_104"
    CORE_TIMEOUT = "CORE_105"
    CORE_CIRCUIT_OPEN = "CORE_106"
    CORE_SERIALIZATION_ERROR = "CORE_107"

    # GRAPH error codes (200-299): Graph operation errors
    GRAPH_NODE_NOT_FOUND = "GRAPH_200"
    GRAPH_EDGE_NOT_FOUND = "GRAPH_201"
    GRAPH_CYCLE_DETECTED = "GRAPH_202"
    GRAPH_INVALID_STRUCTURE = "GRAPH_203"
    GRAPH_OPERATION_FAILED = "GRAPH_204"
    GRAPH_SERIALIZATION_ERROR = "GRAPH_205"
    GRAPH_CONSTRAINT_VIOLATION = "GRAPH_206"
    GRAPH_QUERY_TIMEOUT = "GRAPH_207"

    # SKILL error codes (300-399): Skill execution errors
    SKILL_NOT_FOUND = "SKILL_300"
    SKILL_EXECUTION_FAILED = "SKILL_301"
    SKILL_TIMEOUT = "SKILL_302"
    SKILL_INVALID_INPUT = "SKILL_303"
    SKILL_OUTPUT_VALIDATION_FAILED = "SKILL_304"
    SKILL_RESOURCE_LIMIT_EXCEEDED = "SKILL_305"
    SKILL_DEPENDENCY_ERROR = "SKILL_306"

    # EXTERNAL error codes (400-499): External service errors
    EXTERNAL_SERVICE_UNAVAILABLE = "EXTERNAL_400"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_401"
    EXTERNAL_REQUEST_FAILED = "EXTERNAL_402"
    EXTERNAL_RATE_LIMITED = "EXTERNAL_403"
    EXTERNAL_AUTH_FAILED = "EXTERNAL_404"
    EXTERNAL_TIMEOUT = "EXTERNAL_405"
    EXTERNAL_INVALID_RESPONSE = "EXTERNAL_406"

    # VALIDATION error codes (500-599): Input validation errors
    VALIDATION_FIELD_REQUIRED = "VALIDATION_500"
    VALIDATION_FIELD_INVALID = "VALIDATION_501"
    VALIDATION_TYPE_MISMATCH = "VALIDATION_502"
    VALIDATION_CONSTRAINT_FAILED = "VALIDATION_503"
    VALIDATION_SCHEMA_ERROR = "VALIDATION_504"
    VALIDATION_VALUE_OUT_OF_RANGE = "VALIDATION_505"

    @property
    def category(self) -> ErrorCodeCategory:
        """Extract the category from the error code."""
        prefix = self.value.split("_")[0]
        return ErrorCodeCategory(prefix)

    @property
    def numeric_code(self) -> int:
        """Extract the numeric identifier from the error code."""
        return int(self.value.split("_")[1])

    @property
    def message_template(self) -> str:
        """Get the default message template for this error code."""
        templates: dict[ErrorCode, str] = {
            ErrorCode.CORE_INTERNAL_ERROR: "An internal error occurred",
            ErrorCode.CORE_UNEXPECTED_STATE: "System reached an unexpected state",
            ErrorCode.CORE_CONFIGURATION_ERROR: "Configuration error: {detail}",
            ErrorCode.CORE_DEPENDENCY_MISSING: "Required dependency is missing: {dependency}",
            ErrorCode.CORE_RESOURCE_EXHAUSTED: "Resource exhaustion: {resource}",
            ErrorCode.CORE_TIMEOUT: "Operation timed out after {timeout}s",
            ErrorCode.CORE_CIRCUIT_OPEN: "Circuit breaker is open, operation rejected",
            ErrorCode.CORE_SERIALIZATION_ERROR: "Failed to serialize data: {detail}",
            ErrorCode.GRAPH_NODE_NOT_FOUND: "Graph node not found: {node_id}",
            ErrorCode.GRAPH_EDGE_NOT_FOUND: "Graph edge not found: {source} -> {target}",
            ErrorCode.GRAPH_CYCLE_DETECTED: "Cycle detected in graph traversal",
            ErrorCode.GRAPH_INVALID_STRUCTURE: "Invalid graph structure: {detail}",
            ErrorCode.GRAPH_OPERATION_FAILED: "Graph operation failed: {operation}",
            ErrorCode.GRAPH_SERIALIZATION_ERROR: "Failed to serialize graph: {detail}",
            ErrorCode.GRAPH_CONSTRAINT_VIOLATION: "Graph constraint violated: {constraint}",
            ErrorCode.GRAPH_QUERY_TIMEOUT: "Graph query timed out after {timeout}s",
            ErrorCode.SKILL_NOT_FOUND: "Skill not found: {skill_name}",
            ErrorCode.SKILL_EXECUTION_FAILED: "Skill execution failed: {skill_name}",
            ErrorCode.SKILL_TIMEOUT: "Skill execution timed out: {skill_name}",
            ErrorCode.SKILL_INVALID_INPUT: "Invalid input for skill: {skill_name}",
            ErrorCode.SKILL_OUTPUT_VALIDATION_FAILED: "Skill output validation failed: {skill_name}",
            ErrorCode.SKILL_RESOURCE_LIMIT_EXCEEDED: "Skill resource limit exceeded: {limit}",
            ErrorCode.SKILL_DEPENDENCY_ERROR: "Skill dependency error: {detail}",
            ErrorCode.EXTERNAL_SERVICE_UNAVAILABLE: "External service unavailable: {service}",
            ErrorCode.EXTERNAL_SERVICE_ERROR: "External service error: {service}",
            ErrorCode.EXTERNAL_REQUEST_FAILED: "External request failed: {url}",
            ErrorCode.EXTERNAL_RATE_LIMITED: "Rate limited by external service: {service}",
            ErrorCode.EXTERNAL_AUTH_FAILED: "Authentication failed for external service: {service}",
            ErrorCode.EXTERNAL_TIMEOUT: "External request timed out: {url}",
            ErrorCode.EXTERNAL_INVALID_RESPONSE: "Invalid response from external service: {service}",
            ErrorCode.VALIDATION_FIELD_REQUIRED: "Field is required: {field}",
            ErrorCode.VALIDATION_FIELD_INVALID: "Field has invalid value: {field}",
            ErrorCode.VALIDATION_TYPE_MISMATCH: "Type mismatch for field {field}: expected {expected}",
            ErrorCode.VALIDATION_CONSTRAINT_FAILED: "Constraint validation failed: {constraint}",
            ErrorCode.VALIDATION_SCHEMA_ERROR: "Schema validation error: {detail}",
            ErrorCode.VALIDATION_VALUE_OUT_OF_RANGE: "Value out of range for field {field}",
        }
        return templates.get(self, "Unknown error")

    @property
    def http_status_code(self) -> int | None:
        """Get the recommended HTTP status code for this error."""
        status_map: dict[ErrorCodeCategory, int] = {
            ErrorCodeCategory.CORE: 500,
            ErrorCodeCategory.GRAPH: 400,
            ErrorCodeCategory.SKILL: 400,
            ErrorCodeCategory.EXTERNAL: 502,
            ErrorCodeCategory.VALIDATION: 422,
        }
        return status_map.get(self.category)

    def format_message(self, **kwargs: str) -> str:
        """Format the error message with provided parameters."""
        return self.message_template.format(**kwargs)
