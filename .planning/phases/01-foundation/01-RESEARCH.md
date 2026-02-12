# Phase 1: Foundation - Research Document

**Date:** February 12, 2026
**Objective:** Establish exception hierarchy, structured logging, and config cleanup as base infrastructure for Python 3.13+ codebase

## Executive Summary

This document compiles research findings for implementing Phase 1 of the Graph-RLM infrastructure improvements. The codebase currently uses basic Python logging (`import logging`) with ANSI color formatting, has approximately 100+ broad `except Exception` handlers spread across the codebase, and contains duplicate `LLM_PROVIDER` configuration definitions. The research covers best practices for exception hierarchies, structlog configuration, context preservation, error code patterns, systematic handler replacement strategies, serialization approaches, and config cleanup patterns.

## 1. Python 3.13+ Exception Hierarchy Best Practices

### 1.1 Base Exception Architecture

Python 3.13 introduced several features that enhance exception handling capabilities. The recommended approach for a production-grade exception hierarchy leverages Pydantic models for serialization capabilities while maintaining standard exception inheritance patterns. The base exception class should provide comprehensive context preservation through standard attributes rather than relying on Pydantic model fields, which can complicate standard exception handling patterns.

The core hierarchy should follow a three-tier structure where `BaseGraphRLMError` serves as the ultimate base class, with intermediate category classes for different subsystems, and concrete exception types at the leaf level. This approach mirrors industry standards from projects like Apache Airflow, FastAPI, and LangChain, which all employ similar patterns for maintainable error handling. The intermediate classes should inherit from both `Exception` and potentially a protocol type if runtime checking is needed, while leaf exceptions should be concrete implementations that can be caught by both their specific type and any parent category type.

Python 3.13's improved error messages and exception group handling provide opportunities for more sophisticated error reporting. The new `ExceptionGroup` and `except*` syntax enables handling multiple errors simultaneously, which is particularly useful for concurrent operations. The codebase should leverage these features where async operations might fail with multiple related errors.

### 1.2 Pydantic Integration Strategy

The existing `pydantic-settings` dependency enables a clean approach to exception definition. Rather than making exception classes Pydantic models (which complicates standard exception behavior), the recommended pattern uses Pydantic for configuration-driven error code definitions and handles exception classes as standard Python exceptions with type hints. This separation maintains compatibility with Python's exception mechanism while leveraging Pydantic's validation and serialization capabilities where appropriate.

The error code enum can leverage Pydantic's `StrEnum` or standard `Enum` with string values, providing both type safety and human-readable codes. The enum values should follow a hierarchical naming convention (e.g., `CORE_001`, `GRAPH_101`, `SKILL_201`, `EXTERNAL_301`) that mirrors the exception hierarchy and enables automated error categorization.

### 1.3 Recommended Class Structure

The base exception class should include timestamp generation using `datetime.now(timezone.utc)`, optional correlation_id for request tracing, cause chaining through the standard `__cause__` mechanism, context dictionary for additional metadata, and structured message templates. The class should override `__str__` and `__repr__` to provide developer-friendly output while maintaining the standard exception interface.

Intermediate exception classes (CoreError, GraphError, SkillError, ExternalError) should provide category-specific context handlers and potentially override exception grouping behavior for their domain. Concrete exceptions should be final classes with specific error codes and messages, designed to be caught at the appropriate level of abstraction.

## 2. Structlog Configuration Patterns for Async Code

### 2.1 Structlog Integration Strategy

Structlog provides significant advantages over standard logging for structured data output, which is essential for modern observability platforms. The configuration should prioritize JSON output for production environments while maintaining human-readable output for development. Structlog's processor chain enables adding contextual information (correlation IDs, timestamps, module paths) automatically to every log entry.

For async code, structlog requires careful configuration to ensure proper context management across await boundaries. The recommended approach uses structlog's `copy_context` processors that carry context across async calls, combined with `contextvars` for request-scoped data. This is particularly important for the Graph-RLM codebase, which has extensive async operations in `agent.py`, `runtime.py`, and `kernel.py`.

### 2.2 Processor Chain Configuration

The processor chain should include timestamp formatting as the first processor, adding ISO-formatted timestamps with UTC timezone awareness. Log level detection should follow Python's standard levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) with structlog's corresponding processors. Context enrichment should add correlation IDs, module names, and function names automatically.

For async compatibility, the configuration should use `structlog.contextvars` for thread-local (and async-local) context management. This ensures that when a request correlation ID is set at the entry point of an async function, it propagates through all awaited calls. The `merge_contextvars` processor should be positioned early in the chain to capture this context.

Output formatting should conditionally use different renderers based on environment: `JSONRenderer` for production (log aggregation systems like ELK, Loki, or cloud logging), and `ConsoleRenderer` for development. The configuration should detect the environment through environment variables (e.g., `GRAPH_RLM_ENV=production`) and switch renderers accordingly.

### 2.3 Logger Factory Configuration

Structlog's `configure` or `setup` function should be called during application startup, ideally in the lifespan manager of the FastAPI application (main.py). The configuration should specify processors for both the `structlog` logger and the `logging` compatibility layer.

The `logging` compatibility layer is critical for this codebase because many dependencies (FalkorDB, LangChain, FastAPI) use standard logging. The configuration should wrap standard loggers with structlog processors to ensure consistent output format across all log entries. This is achieved through `structlog.stdlib.add_log_level`, `structlog.stdlib.ProcessorFormatter`, and configuring the standard `logging` handlers to use structlog formatters.

## 3. Exception Context Preservation

### 3.1 Correlation ID Implementation

Correlation IDs (also known as trace IDs or request IDs) are essential for tracing errors across the distributed components of the Graph-RLM system. The implementation should generate correlation IDs at the request boundary (FastAPI middleware) and propagate them through all subsequent operations, including async tasks.

The recommended implementation uses `contextvars` for correlation ID storage, which provides async-safety out of the box in Python 3.7+. The middleware should generate a UUID if none is provided in the request headers (following W3C Trace Context standard if available), set it in the context variable, and add it to all log entries through structlog processors. Exception handlers should automatically capture the correlation ID from the context variable and include it in the exception's context dictionary.

### 3.2 Cause Chaining Pattern

Python's exception chaining mechanism (`raise ... from ...`) preserves the causal relationship between exceptions while maintaining clean exception hierarchies. The implementation should follow these patterns: wrap lower-level exceptions (database errors, network errors) in higher-level exceptions (GraphError, SkillExecutionError) while preserving the original exception as the `__cause__` attribute.

The base exception class should provide a `wrap_cause` method that simplifies this pattern. When catching a lower-level exception and raising a higher-level one, the method should automatically set the cause, copy relevant context from the cause to the new exception, and log both appropriately. This ensures that exception chains are properly preserved for debugging while presenting clean, category-specific exceptions to calling code.

### 3.3 Context Dictionary Structure

Each exception should maintain a context dictionary containing standard fields (correlation_id, timestamp, error_code, module, function, line_number) plus domain-specific fields (graph_operation, skill_name, external_service, etc.). The structure should be designed for easy serialization to JSON for API responses and log aggregation systems.

The context dictionary should support nested structures for complex errors (e.g., validation errors with multiple field failures, aggregation errors with multiple cause exceptions). A `context` property should provide a frozen view of the context dictionary to prevent accidental modification after exception construction.

## 4. Error Code Enum Patterns

### 4.1 Hierarchical Category Structure

The error code enum should follow a hierarchical structure that mirrors the exception hierarchy. Each category (CORE, GRAPH, SKILL, EXTERNAL) should have its own numeric range for easy categorization:

- `CORE_000` - `CORE_099`: Core system errors (configuration, initialization, shutdown)
- `GRAPH_100` - `GRAPH_199`: Graph database and traversal errors
- `SKILL_200` - `SKILL_299`: Skill execution and lifecycle errors
- `EXTERNAL_300` - `EXTERNAL_399`: External service integration errors
- `VALIDATION_400` - `VALIDATION_499`: Data validation errors

This structure enables quick error categorization from error codes alone and provides natural separation for documentation and error tracking dashboards.

### 4.2 Enum Member Design

Each enum member should be a string value containing the code for easy serialization and logging. The enum class should provide properties for category extraction (e.g., `error_code.category` returning `"GRAPH"`) and numeric range access. A method should convert enum members to human-readable descriptions.

The recommended pattern uses `str.Enum` with string values:

```python
from enum import Enum

class ErrorCode(Enum):
    CORE_001 = "CORE_001"  # "Configuration file not found"
    GRAPH_101 = "GRAPH_101"  # "Graph connection failed"
    SKILL_201 = "SKILL_201"  # "Skill execution timeout"
    EXTERNAL_301 = "EXTERNAL_301"  # "External API rate limited"
```

Additional properties can be added for categorization and human-readable descriptions:

```python
@property
def category(self) -> str:
    return self.value.split("_")[0]

@property
def number(self) -> int:
    return int(self.value.split("_")[1])
```

### 4.3 Integration with Exceptions

Exception classes should accept error codes as their primary constructor argument, validating the code against the enum and extracting the default message. The exception should also accept an optional override message for cases where the standard message doesn't fit the specific error scenario.

The pattern should be:

```python
class GraphError(BaseGraphRLMError):
    def __init__(
        self,
        error_code: ErrorCode,
        message: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **context
    ):
        self.error_code = error_code
        super().__init__(
            message=message or error_code.description,
            correlation_id=correlation_id,
            **context
        )
```

## 5. Systematic Broad Exception Handler Replacement

### 5.1 Discovery Strategy

The codebase contains approximately 100 `except Exception` handlers that need systematic replacement. The strategy should prioritize handlers by risk level and impact:

1. High Priority: Handlers in critical paths (database operations, external API calls, agent execution)
2. Medium Priority: Handlers in skill execution and MCP integration code
3. Low Priority: Handlers in initialization and utility code

A grep-based approach can categorize handlers by file:

```bash
# Find all broad exception handlers with context
grep -rn "except Exception" --include="*.py" graph_rlm/ | \
  grep -v "# pylint: disable" | \
  grep -v "# noqa:" | \
  head -50
```

### 5.2 Categorization Framework

Each handler should be analyzed to determine the specific exception types it should catch. Common categories include:

- **IO Operations**: `IOError`, `OSError`, `FileNotFoundError`
- **Network Operations**: `requests.RequestException`, `aiohttp.ClientError`
- **Database Operations**: `falkordb.Error`, `redis.RedisError`
- **Validation**: `ValueError`, `TypeError`, `pydantic.ValidationError`
- **Concurrency**: ` asyncio.CancelledError`, `TimeoutError`

The handler should catch the narrowest possible exception type(s), with the specific Graph-RLM exception types as the outer wrapper for error categorization and logging.

### 5.3 Refactoring Pattern

The standard refactoring pattern should follow this structure:

```python
# BEFORE:
try:
    operation()
except Exception as e:
    logger.error("Operation failed: %s", e)
    return f"Error: {e}"

# AFTER:
try:
    operation()
except SpecificError as e:
    logger.error("Operation failed: %s", e, extra={"error_code": "CORE_001"})
    raise GraphError(
        error_code=ErrorCode.GRAPH_101,
        message="Failed to perform graph operation",
        cause=e,
        operation="graph_traverse"
    ) from e
except Exception as e:
    logger.error("Unexpected error: %s", e)
    raise BaseGraphRLMError(
        error_code=ErrorCode.CORE_000,
        message="Unexpected error",
        cause=e
    ) from e
```

### 5.4 Testing Strategy

Each refactored handler should have corresponding tests that verify:
1. Specific exceptions are caught by the appropriate handlers
2. Exception context (correlation_id, error_code, cause chain) is preserved
3. Logging output is correctly structured
4. API responses contain the expected error format

A test helper function can validate exception hierarchy and context:

```python
def assert_exception_context(
    exc: BaseGraphRLMError,
    expected_code: ErrorCode,
    expected_correlation_id: Optional[str] = None
):
    assert isinstance(exc.error_code, ErrorCode)
    assert exc.error_code == expected_code
    if expected_correlation_id:
        assert exc.correlation_id == expected_correlation_id
    assert isinstance(exc.timestamp, datetime)
```

## 6. Exception-to-Dict Serialization for API Responses

### 6.1 Serialization Design

FastAPI needs to return structured error responses that include error codes, messages, correlation IDs, and context. The serialization should produce JSON-compatible dictionaries that follow a consistent schema across all error types.

The recommended schema:

```python
{
    "error": {
        "code": "GRAPH_101",
        "message": "Human-readable error message",
        "category": "GRAPH",
        "timestamp": "2026-02-12T10:30:00Z",
        "correlation_id": "550e8400-e29b-41d4-a716-446655440000"
    },
    "context": {
        "operation": "graph_traverse",
        "node_id": "12345"
    },
    "cause": {  // Optional, recursive structure
        "error": {
            "code": "CORE_000",
            "message": "Original error message"
        }
    }
}
```

### 6.2 Serializer Implementation

The base exception class should implement `to_dict()` and `to_json()` methods:

```python
def to_dict(self, include_cause: bool = True) -> dict:
    result = {
        "error": {
            "code": self.error_code.value if self.error_code else None,
            "message": str(self.message),
            "category": self.error_code.category if self.error_code else None,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id
        },
        "context": dict(self.context)
    }
    if self.__cause__ and include_cause:
        if hasattr(self.__cause__, "to_dict"):
            result["cause"] = self.__cause__.to_dict(include_cause=False)
        else:
            result["cause"] = {
                "error": {
                    "message": str(self.__cause__),
                    "type": type(self.__cause__).__name__
                }
            }
    return result
```

### 6.3 FastAPI Exception Handler

The FastAPI application should include a custom exception handler that serializes Graph-RLM exceptions and handles unknown exceptions gracefully:

```python
from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse

@app.exception_handler(BaseGraphRLMError)
async def graph_rlm_exception_handler(request: Request, exc: BaseGraphRLMError):
    return JSONResponse(
        status_code=determine_http_status(exc),
        content=exc.to_dict()
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "CORE_000",
                "message": "Internal server error",
                "correlation_id": get_correlation_id()
            }
        }
    )
```

The HTTP status code mapping should be defined based on error categories:
- `CORE_*`: 500 for internal errors, 400 for bad requests
- `GRAPH_*`: 500 for database errors, 404 for not found
- `SKILL_*`: 500 for execution errors, 400 for invalid input
- `EXTERNAL_*`: 502 for gateway errors, 429 for rate limiting
- `VALIDATION_*`: 422 for unprocessable entity, 400 for bad request

## 7. Config Cleanup Patterns

### 7.1 Duplicate LLM_PROVIDER Resolution

The codebase currently has duplicate `LLM_PROVIDER` definitions:

1. **config.py line 38**: `LLM_PROVIDER: str = "openrouter"` (in Settings class)
2. **config.py line 46**: `LLM_PROVIDER: str = "openrouter"` (duplicate in same class)
3. **llm.py line 73**: `return settings.LLM_PROVIDER`
4. **endpoints.py lines 74, 84**: References to `settings.LLM_PROVIDER`
5. **mcp_integration/config.py line 123**: `LLM_PROVIDER=ollama` (MCP environment variable)

The resolution strategy should:

1. **Consolidate to Single Definition**: Remove line 46 and keep line 38 as the canonical definition in `config.py`
2. **Update MCP Integration**: Ensure `mcp_integration/config.py` reads from the central settings rather than duplicating the environment variable
3. **Documentation**: Document the canonical location for any future configuration additions

### 7.2 Environment Variable Precedence

Pydantic Settings follows this precedence order (highest to lowest):
1. Environment variables
2. .env file
3. Secrets (if configured)
4. Code defaults

This means the duplicate definition in the class doesn't create problems, but it creates confusion and maintenance burden. The canonical definition at line 38 should remain, with the duplicate at line 46 removed.

### 7.3 MCP Config Integration

The MCP integration code has a different configuration mechanism. The resolution should:

1. Create a wrapper in `mcp_integration/config.py` that reads from central settings
2. Remove hardcoded `LLM_PROVIDER` from MCP environment configuration
3. Ensure backward compatibility by checking `MCP_LLM_PROVIDER` environment variable as fallback

```python
# In mcp_integration/config.py
from graph_rlm.backend.src.core.config import settings

def get_mcp_llm_provider() -> str:
    """Get LLM provider for MCP integration, with MCP-specific override."""
    import os
    return os.getenv("MCP_LLM_PROVIDER", settings.LLM_PROVIDER)
```

## 8. Type Hints for Exception Classes

### 8.1 Comprehensive Type Annotation Strategy

Python 3.13's improved type checking capabilities enable more expressive type annotations for exception classes. The type hints should cover:

- **Exception attributes**: `message: str`, `correlation_id: Optional[str]`
- **Context dictionary**: `context: dict[str, Any]`
- **Cause chain**: `__cause__: Optional[BaseException]`
- **Error code**: `error_code: Optional[ErrorCode]`
- **Return type for conversion methods**: `to_dict() -> dict[str, Any]`

### 8.2 Generic Exception Handling

For exception handlers that catch and re-raise, the type hints should enable precise type narrowing:

```python
def handle_operation() -> ResultType:
    try:
        return perform_operation()
    except BaseGraphRLMError as exc:
        # Type narrowing: exc is known to be BaseGraphRLMError here
        logger.error(
            "Operation failed",
            error_code=exc.error_code.value,
            correlation_id=exc.correlation_id
        )
        raise
```

### 8.3 Protocol for Exception Types

Defining a protocol for exception handlers enables flexible handler signatures:

```python
from typing import Protocol

class ExceptionHandler(Protocol):
    def __call__(self, exc: BaseGraphRLMError) -> None:
        ...
```

This enables standardized exception handler registration across different subsystems.

## 9. Implementation Roadmap

### Phase 1A: Foundation Classes (Week 1)

- Create `exceptions/__init__.py` with base classes
- Implement ErrorCode enum with 50+ common error codes
- Create structured logging configuration
- Add exception-to-dict serialization methods
- Write unit tests for exception hierarchy

### Phase 1B: Exception Handler Migration (Week 2)

- Categorize 100+ exception handlers by risk level
- Refactor high-priority handlers first (database, external APIs)
- Update skill execution handlers
- Update MCP integration handlers
- Add integration tests for handler behavior

### Phase 1C: Config Cleanup and Documentation (Week 3)

- Remove duplicate LLM_PROVIDER definitions
- Update MCP integration configuration
- Document error code patterns
- Create exception handling best practices guide
- Update contributing guidelines

### Phase 1D: Validation and Optimization (Week 4)

- Performance testing of exception handling path
- Memory profiling for exception context storage
- Load testing with correlated request IDs
- Logging volume analysis and optimization
- Final documentation review

## 10. Key Dependencies and Additions

The implementation requires adding `structlog` to the project dependencies:

```toml
# In pyproject.toml
[dependencies]
# ... existing dependencies ...
structlog = ">=24.0.0"
```

No other new dependencies are required; the implementation uses only standard library features and existing Pydantic functionality.

## 11. Risks and Mitigations

### 11.1 Risk: Breaking Changes

**Risk**: Refactoring exception handlers might change behavior and break existing code paths.

**Mitigation**: 
- Maintain backward compatibility by catching old exception types in addition to new ones during transition
- Add deprecation warnings for old patterns
- Implement comprehensive test coverage before refactoring
- Use feature flags to enable/disable new exception handling

### 11.2 Risk: Performance Impact

**Risk**: Structured logging and exception context preservation might add overhead.

**Mitigation**:
- Use lazy evaluation for expensive context fields
- Benchmark current performance before changes
- Optimize serialization paths
- Consider conditional context inclusion based on log level

### 11.3 Risk: Logging Volume Increase

**Risk**: More structured logging might increase log volume significantly.

**Mitigation**:
- Configure appropriate log levels for production
- Implement log sampling for high-volume DEBUG logs
- Use log filtering based on error codes
- Monitor and adjust based on actual usage

## 12. References and Resources

### 12.1 Python Documentation

- Python 3.13 Exception Groups: https://docs.python.org/3/library/exceptions.html
- contextvars Module: https://docs.python.org/3/library/contextvars.html
- typing module extensions for 3.13: https://docs.python.org/3/library/typing.html

### 12.2 Structlog Documentation

- Structlog Getting Started: https://structlog.readthedocs.io/en/stable/getting-started.html
- Processor Chain Configuration: https://structlog.readthedocs.io/en/stable/processors.html
- Async Compatibility: https://structlog.readthedocs.io/en/stable/thread-local.html

### 12.3 Pydantic Patterns

- Pydantic Settings: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
- StrEnum Pattern: https://docs.pydantic.dev/latest/api/std_enum/

### 12.4 Industry Examples

- FastAPI Exception Handling: https://fastapi.tiangolo.com/tutorial/handling-errors/
- Apache Airflow Exception Patterns: https://airflow.apache.org/docs/apache-airflow/stable/concepts/tasks.html
- LangChain Error Handling: https://python.langchain.com/docs/concepts/errors/

---

**Document Version:** 1.0
**Last Updated:** February 12, 2026
**Next Review:** Before Phase 1 implementation begins
