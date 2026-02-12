---
wave: 0
depends_on: []
autonomous: true
files_modified: []
---

# Phase 1: Foundation - Master Plan Index

## Overview

This document serves as the master index for all Phase 1 execution plans. Each sub-plan addresses specific requirements with clear wave assignments and dependencies.

## Phase Goal

Establish exception hierarchy, structured logging, and config cleanup as base infrastructure.

## Success Criteria

1. Developer can create `BaseGraphRLMError` subclass with error code, correlation_id, and cause chaining preserved
2. Developer can import specific exception types from `src/core/exceptions`
3. Developer can call `error.to_dict()` on any exception to get serializable API response format
4. All exception handlers use specific types instead of broad `except Exception`
5. Developer can import configured structlog logger producing JSON-structured output with enriched exception context

## Wave Structure

### Wave 1: Exception Foundation (Independent)
- **plan-01-exception-base.md**: Base classes, ErrorCode, specific types
- **plan-05-config-cleanup.md**: Remove duplicate LLM_PROVIDER
- **Duration**: Week 1

### Wave 2: Logging & Serialization (Depends on Wave 1)
- **plan-02-logging-config.md**: structlog configuration
- **plan-03-serialization.md**: Exception serialization (depends on plan-01)
- **Duration**: Week 1-2

### Wave 3: Handler Migration (Depends on Waves 1-2)
- **plan-04-handler-migration.md**: Replace 141+ broad handlers
- **Duration**: Week 2-3

## Plan Files

| Plan | File | Requirements | Wave | Status |
|------|------|--------------|------|--------|
| 01 | plan-01-exception-base.md | EXCP-01, EXCP-02, EXCP-03, REFR-03 | 1 | Ready |
| 02 | plan-02-logging-config.md | LOG-01, LOG-02 | 2 | Ready |
| 03 | plan-03-serialization.md | EXCP-06 | 2 | Ready |
| 04 | plan-04-handler-migration.md | EXCP-04, EXCP-05 | 3 | Ready |
| 05 | plan-05-config-cleanup.md | REFR-01 | 1 | Ready |

## Execution Order

### Week 1
1. Execute plan-01-exception-base.md (Foundation exception classes)
2. Execute plan-05-config-cleanup.md (Config cleanup - quick win)
3. Begin plan-02-logging-config.md (Logging setup)

### Week 2
1. Complete plan-02-logging-config.md (Logging configuration)
2. Execute plan-03-serialization.md (Exception serialization)
3. Begin plan-04-handler-migration.md (Handler migration - high priority)

### Week 3
1. Complete plan-04-handler-migration.md (Handler migration)
2. Phase 1 validation and review

## Key Implementation Decisions

### 1. Exception Class Design
- BaseGraphRLMError uses standard Exception inheritance, not Pydantic model
- ErrorCode enum uses string values for easy serialization
- Context stored in dict for flexibility, exposed via property

### 2. Logging Configuration
- structlog for structured JSON output
- ConsoleRenderer for development, JSONRenderer for production
- contextvars for correlation ID propagation

### 3. Handler Migration Pattern
- Replace broad `except Exception` with specific types
- Use `raise ... from e` for cause chain preservation
- Log with structlog before raising

## Requirements Coverage

| Requirement | Plan | Status |
|-------------|------|--------|
| EXCP-01 | plan-01 | Pending |
| EXCP-02 | plan-01 | Pending |
| EXCP-03 | plan-01 | Pending |
| EXCP-04 | plan-04 | Pending |
| EXCP-05 | plan-04 | Pending |
| EXCP-06 | plan-03 | Pending |
| REFR-01 | plan-05 | Pending |
| REFR-03 | plan-01 | Pending |
| LOG-01 | plan-02 | Pending |
| LOG-02 | plan-02 | Pending |

## Exit Criteria (Phase 1 Complete)

1. All 10 Phase 1 requirements implemented
2. All success criteria validated
3. 100% of except Exception handlers replaced (EXCP-04)
4. Structured logging configured and producing JSON output (LOG-01, LOG-02)
5. Duplicate LLM_PROVIDER removed (REFR-01)
6. No new exception handler regressions
