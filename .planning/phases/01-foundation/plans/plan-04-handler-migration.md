---
wave: 3
depends_on: ["plan-01-exception-base", "plan-02-logging-config"]
autonomous: true
files_modified:
  - graph_rlm/backend/src/core/exceptions/handlers.py (new)
  - All Python files with except Exception handlers (to be identified)
---

# Phase 1: Foundation - Exception Handler Migration

## Overview

Replace 141+ broad `except Exception` handlers with specific exception types and add structured logging. This is the most extensive refactoring task in Phase 1.

## Requirements Addressed

- EXCP-04: Replace 141+ broad exception handlers with specific types
- EXCP-05: Structured logging in all exception handlers

## Implementation Order

1. Discover and categorize all exception handlers
2. Create exception handler utilities
3. Migrate high-priority handlers (database, external APIs)
4. Migrate medium-priority handlers (skill execution, MCP)
5. Migrate low-priority handlers (initialization, utilities)
6. Verify all migrations

## Tasks

### 4.1 Discover and categorize handlers

Run discovery command:
```bash
grep -rn "except Exception" --include="*.py" graph_rlm/ | \
  grep -v "# pylint: disable" | \
  grep -v "# noqa:" | \
  grep -v "tests/" > /tmp/exception_handlers.csv
```

Categorize by:
- HIGH: Database operations, external API calls, agent execution
- MEDIUM: Skill execution, MCP integration
- LOW: Initialization, utilities, one-time operations

### 4.2 Create handler utilities (handlers.py)

```python
"""Exception handler utilities for consistent error handling."""

from typing import Any, TypeVar

from structlog import get_logger

from .base import BaseGraphRLMError
from .codes import ErrorCode
from .types import (
    CoreError,
    GraphError,
    SkillExecutionError,
    ExternalServiceError,
    ValidationError,
)
from ..logging import get_logger, get_correlation_id

T = TypeVar("T", bound=BaseGraphRLMError)

class ExceptionHandler:
    """Standardized exception handler with logging and context."""
    
    def __init__(self, *, log_level: str = "error"):
        self.logger = get_logger(__name__)
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
            log_message: Custom log message (default: "Exception occurred")
        
        Returns:
            The exception (possibly wrapped) if not reraised
        """
        correlation_id = get_correlation_id()
        
        # Build context
        handler_context: dict[str, Any] = {
            "error_code": error_code.value,
            "correlation_id": correlation_id,
            "exception_type": type(exception).__name__,
        }
        if context:
            handler_context.update(context)
        
        # Log the exception
        log_msg = log_message or f"{type(exception).__name__}: {str(exception)}"
        log_method = getattr(self.logger, self.log_level)
        log_method(log_msg, exc_info=True, **handler_context)
        
        # Build and potentially raise the new exception
        if reraise_as:
            raise reraise_as(
                message=message or str(exception),
                error_code=error_code,
                correlation_id=correlation_id,
                cause=exception,
                **(context or {}),
            ) from exception
        
        return None
```

### 4.3 Migrate high-priority handlers

Focus on files with database and external API operations:
- src/core/config.py (except Exception at line 170)
- src/core/llm.py (LLM service calls)
- src/integrations/ (external API integrations)

Pattern:
```python
# BEFORE:
try:
    operation()
except Exception as e:
    logger.error("Operation failed: %s", e)
    return f"Error: {e}"

# AFTER:
from .exceptions import GraphError, ErrorCode
from .logging import get_logger

try:
    operation()
except SpecificError as e:
    logger.error("Operation failed", exc_info=True, error_code="GRAPH_103")
    raise GraphError(
        error_code=ErrorCode.GRAPH_103,
        message="Failed to perform graph operation",
        cause=e,
        operation="graph_traverse"
    ) from e
except Exception as e:
    logger.error("Unexpected error", exc_info=True, error_code="CORE_000")
    raise CoreError(
        error_code=ErrorCode.CORE_000,
        message="Unexpected error during operation",
        cause=e,
    ) from e
```

### 4.4 Migrate medium-priority handlers

Focus on skill execution and MCP integration:
- src/skills/ (skill execution)
- mcp_integration/ (MCP server connections)

### 4.5 Migrate low-priority handlers

Focus on initialization and utility code:
- src/main.py (app startup)
- src/core/ (core utilities)

## Verification Criteria

- [ ] 100% of except Exception handlers replaced with specific types
- [ ] Each handler catches the narrowest possible exception type
- [ ] All handlers use Graph-RLM exception types as outer wrapper
- [ ] All handlers include structured logging via structlog
- [ ] All handlers preserve cause chain via `raise ... from e`
- [ ] Error codes are appropriate for the operation type
- [ ] No new broad except Exception blocks remain (except final fallback)

## Must Haves (Goal-Backward Verification)

1. **EXCP-04 satisfied**: All exception handlers in the codebase use specific types instead of broad except Exception
2. **EXCP-05 satisfied**: All exception handlers include structured logging with full context (error_code, correlation_id, exception_type)
3. Exception cause chains are preserved through raise ... from e
4. Error codes map correctly to exception categories
