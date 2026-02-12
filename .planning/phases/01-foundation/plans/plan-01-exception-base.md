---
wave: 1
depends_on: []
autonomous: true
files_modified:
  - graph_rlm/backend/src/core/exceptions/__init__.py (new)
  - graph_rlm/backend/src/core/exceptions/base.py (new)
  - graph_rlm/backend/src/core/exceptions/codes.py (new)
  - graph_rlm/backend/src/core/exceptions/types.py (new)
  - pyproject.toml (structlog dependency)
---

# Phase 1: Foundation - Exception Base Classes

## Overview

Implement the exception hierarchy foundation including BaseGraphRLMError, ErrorCode enum, and specific exception types. This creates the base infrastructure all subsequent exception handling depends on.

## Requirements Addressed

- EXCP-01: Base exception class with context preservation (correlation_id, timestamp, cause chaining)
- EXCP-02: ErrorCode enum with hierarchical categories (CORE_*, GRAPH_*, SKILL_*, EXTERNAL_*)
- EXCP-03: Specific exception types (GraphError, SkillExecutionError, ExternalServiceError, ValidationError)
- REFR-03: Type hints for exception classes

## Implementation Order

1. ErrorCode enum with hierarchical categories
2. BaseGraphRLMError base class
3. Intermediate exception categories
4. Specific exception types
5. Public API exports in __init__.py

## Tasks

### 1.1 Create ErrorCode Enum (codes.py)

Create ErrorCode enum with 20+ error codes across 5 hierarchical categories (CORE, GRAPH, SKILL, EXTERNAL, VALIDATION). Each code should be a string value (e.g., "GRAPH_101") with properties for category extraction and numeric range.

### 1.2 Create BaseGraphRLMError (base.py)

Create base exception class with:
- message, error_code, correlation_id, timestamp (UTC)
- **context for arbitrary metadata
- __cause__ preservation for exception chaining
- with_context() and with_correlation_id() chaining methods
- __str__ and __repr__ implementations
- source_location property

### 1.3 Create Specific Exception Types (types.py)

Create 5 exception types:
- CoreError (with_operation helper)
- GraphError (with_graph_operation, with_node_id helpers)
- SkillExecutionError (with_skill_name, with_skill_input helpers)
- ExternalServiceError (with_service_name, with_endpoint helpers)
- ValidationError (with_field_errors, with_schema helpers)

### 1.4 Create Public API Exports (__init__.py)

Export all exception classes and ErrorCode for public use.

### 1.5 Update pyproject.toml for structlog

Add structlog dependency to pyproject.toml.

## Verification Criteria

- [ ] ErrorCode enum with 20+ error codes across 5 categories
- [ ] BaseGraphRLMError accepts all required parameters
- [ ] with_context() and with_correlation_id() chain correctly
- [ ] __cause__ preserves exception chain
- [ ] Specific exception types inherit from BaseGraphRLMError
- [ ] All type hints follow Python 3.13+ standards
- [ ] Import from src.core.exceptions works correctly

## Must Haves (Goal-Backward Verification)

1. **EXCP-01 satisfied**: BaseGraphRLMError exists with correlation_id, timestamp, cause chaining
2. **EXCP-02 satisfied**: ErrorCode enum with hierarchical categories
3. **EXCP-03 satisfied**: Specific exception types importable from src/core/exceptions
4. **REFR-03 satisfied**: All exception classes have comprehensive type hints
