# Graph-RLM Engineering Health Roadmap

**Created:** 2026-02-12  
**Depth:** Comprehensive  
**Confidence:** HIGH

## Overview

Engineering health improvements for Graph-RLM focusing on exception handling, test infrastructure, and circuit breaker patterns. The roadmap implements 29 v1 requirements across 5 phases, building from foundational infrastructure through to full business logic integration.

## Phase Summary

| Phase | Goal | Plans | Requirements | Success Criteria |
|-------|------|-------|--------------|------------------|
| 1 - Foundation | Exception hierarchy, logging infrastructure, config cleanup | 5 plans | 9 | 5 |
| 2 - Core Abstractions | Async circuit breaker pattern implementation | 3 plans | 7 | 5 |
| 3 - Test Infrastructure | pytest setup, mocking, initial unit tests | — | 6 | 6 |
| 4 - Business Logic Integration | Agent circuit breaker integration, error handling | — | 4 | 4 |
| 5 - API and Integration | External service integration, coverage validation | — | 3 | 4 |

---

## Phase 1: Foundation

**Goal:** Establish exception hierarchy, structured logging, and config cleanup as base infrastructure.

**Dependencies:** None (foundational)

**Requirements:**
- EXCP-01: Base exception class with context preservation
- EXCP-02: ErrorCode enum with hierarchical categories
- EXCP-03: Specific exception types (GraphError, SkillExecutionError, ExternalServiceError, ValidationError)
- EXCP-04: Replace 141+ broad exception handlers
- EXCP-05: Structured logging in all handlers
- EXCP-06: Exception-to-dict serialization
- REFR-01: Remove duplicate LLM_PROVIDER config
- REFR-03: Type hints for exception classes
- LOG-01: structlog configuration for JSON logging
- LOG-02: Exception context enrichment

**Success Criteria:**
1. Developer can create a `BaseGraphRLMError` subclass with error code, correlation_id, and cause chaining preserved
2. Developer can import specific exception types (GraphError, SkillExecutionError, ExternalServiceError, ValidationError) from src/core/exceptions
3. Developer can call `error.to_dict()` on any exception to get serializable API response format
4. All exception handlers in the codebase use specific exception types instead of broad `except Exception`
5. Developer can import configured structlog logger that produces JSON-structured output with enriched exception context

---

## Phase 2: Core Abstractions

**Goal:** Implement async-aware circuit breaker pattern for resilience against external service failures.

**Dependencies:** Phase 1 complete (exception hierarchy and logging required for circuit state transition logging)

**Requirements:**
- CIRCB-01: CircuitState enum (CLOSED, OPEN, HALF_OPEN)
- CIRCB-02: CircuitBreakerConfig dataclass
- CIRCB-03: Async-aware CircuitBreaker class with state machine
- CIRCB-04: CircuitOpenError exception
- CIRCB-05: LLM service call integration
- CIRCB-06: MCP server connection integration
- CIRCB-07: Metrics/observability hooks
- LOG-03: Correlation ID propagation

**Plans:**
- [02-01-PLAN.md](./phases/02-core-abstractions/02-01-PLAN.md) — Core circuit breaker infrastructure (CircuitState, CircuitBreakerConfig, CircuitBreaker class, CircuitOpenError)
- [02-02-PLAN.md](./phases/02-core-abstractions/02-02-PLAN.md) — LLM service integration with circuit breaker and correlation ID propagation
- [02-03-PLAN.md](./phases/02-core-abstractions/02-03-PLAN.md) — MCP server integration with metrics/observability hooks

**Success Criteria:**
1. Developer can instantiate CircuitBreaker with configurable failure_threshold, timeout, success_threshold
2. Developer can observe circuit state transitions (CLOSED→OPEN→HALF_OPEN→CLOSED) logged via structlog
3. Developer can catch CircuitOpenError when circuit is OPEN and receive proper exception with context
4. Developer can call `circuit_breaker.callAsync(llm_service.query, prompt)` and have it protect against cascading failures
5. Developer can observe correlation_id propagated through async call chain via logging context

---

## Phase 3: Test Infrastructure

**Goal:** Establish pytest configuration, mock registry, and initial unit tests for core modules.

**Dependencies:** Phase 1 complete (exceptions module must exist for TEST-06)

**Requirements:**
- TEST-01: pytest with asyncio_mode=auto configuration
- TEST-02: Mock registry (tests/mocking/mocks.py) for FalkorDB, LLM service, external APIs
- TEST-03: Base test fixtures for async setup/teardown
- TEST-04: MockRegistry class with reset capability
- TEST-05: Unit tests for src/core/config.py (100% coverage)
- TEST-06: Unit tests for src/core/exceptions/base.py (100% coverage)
- TEST-07: pytest-cov configuration for incremental coverage
- REFR-02: Isolated test module for src/core/agent.py with legacy mocking

**Success Criteria:**
1. Developer can run `pytest` and have all tests in tests/ directory execute with asyncio support
2. Developer can access MockRegistry from tests to create FalkorDB, LLM service, and HTTP mocks
3. Developer can use async fixtures that properly clean up after each test
4. Developer can call `mock_registry.reset()` to clear state between test runs
5. Developer can achieve 100% coverage on src/core/config.py
6. Developer can achieve 100% coverage on src/core/exceptions/base.py
7. Developer can write unit tests for agent.py using legacy mocking patterns

**Plans:**
- [03-01-PLAN.md](./phases/03-test-infrastructure/03-01-PLAN.md) — pytest configuration + MockRegistry class foundation
- [03-02-PLAN.md](./phases/03-test-infrastructure/03-02-PLAN.md) — FalkorDB/LLM/external mocks + enhanced fixtures
- [03-03-PLAN.md](./phases/03-test-infrastructure/03-03-PLAN.md) — Unit tests for config.py, exceptions/base.py, agent.py

 ---

## Phase 4: Business Logic Integration

**Goal:** Integrate circuit breakers and error handling into agent core with graceful degradation.

**Dependencies:** Phase 2 and Phase 3 complete (circuit breaker must exist, tests must be in place)

**Requirements:**
- (Circuit breaker integration into agent core - implicit from Phase 2 CIRCB-05)
- (Error logging integration - implicit from Phase 1 EXCP-05)
- (Validation patterns - implicit from EXCP-03 ValidationError)

**Success Criteria:**
1. Developer can observe agent.main_loop operating with circuit breaker protection (LLM calls fail gracefully when circuit opens)
2. Developer can observe structured error logs with full context when agent encounters failures
3. Agent can gracefully degrade when external services (LLM, MCP) become unavailable
4. Developer can verify input validation using ValidationError with proper error codes

---

## Phase 5: API and Integration

**Goal:** Complete external service integration and validate comprehensive coverage.

**Dependencies:** Phase 4 complete (business logic must be protected)

**Requirements:**
- (MCP server integration validation - implicit from Phase 2 CIRCB-06)
- (Comprehensive coverage validation - implicit from Phase 3 TEST-07)
- (Config cleanup validation - implicit from Phase 1 REFR-01)

**Success Criteria:**
1. Developer can verify MCP server connections are protected by circuit breaker (connection failures handled gracefully)
2. Developer can run coverage report and see percentage for new infrastructure code
3. Developer can verify duplicate LLM_PROVIDER config has been removed from Settings class
4. Developer can run full test suite and confirm all new infrastructure tests pass

---

## Traceability

### Requirements → Phase Mapping

| Requirement | Phase | Category | Description |
|-------------|-------|----------|-------------|
| EXCP-01 | Phase 1 | Exception | Base exception class with context preservation |
| EXCP-02 | Phase 1 | Exception | ErrorCode enum with hierarchical categories |
| EXCP-03 | Phase 1 | Exception | Specific exception types |
| EXCP-04 | Phase 1 | Exception | Replace 141+ broad exception handlers |
| EXCP-05 | Phase 1 | Exception | Structured logging in handlers |
| EXCP-06 | Phase 1 | Exception | Exception-to-dict serialization |
| CIRCB-01 | Phase 2 | CircuitBreaker | CircuitState enum |
| CIRCB-02 | Phase 2 | CircuitBreaker | CircuitBreakerConfig dataclass |
| CIRCB-03 | Phase 2 | CircuitBreaker | Async-aware CircuitBreaker class |
| CIRCB-04 | Phase 2 | CircuitBreaker | CircuitOpenError exception |
| CIRCB-05 | Phase 2 | CircuitBreaker | LLM service integration |
| CIRCB-06 | Phase 2 | CircuitBreaker | MCP server integration |
| CIRCB-07 | Phase 2 | CircuitBreaker | Metrics/observability hooks |
| TEST-01 | Phase 3 | Test | pytest configuration |
| TEST-02 | Phase 3 | Test | Mock registry |
| TEST-03 | Phase 3 | Test | Base fixtures |
| TEST-04 | Phase 3 | Test | MockRegistry class |
| TEST-05 | Phase 3 | Test | config.py unit tests |
| TEST-06 | Phase 3 | Test | exceptions/base.py unit tests |
| TEST-07 | Phase 3 | Test | pytest-cov configuration |
| REFR-01 | Phase 1 | Refactor | Remove duplicate LLM_PROVIDER |
| REFR-02 | Phase 3 | Refactor | agent.py test module |
| REFR-03 | Phase 1 | Refactor | Type hints for exceptions |
| LOG-01 | Phase 1 | Logging | structlog configuration |
| LOG-02 | Phase 1 | Logging | Exception context enrichment |
| LOG-03 | Phase 2 | Logging | Correlation ID propagation |

### Phase → Requirements Summary

| Phase | Requirements | Count |
|-------|--------------|-------|
| Phase 1 | EXCP-01, EXCP-02, EXCP-03, EXCP-04, EXCP-05, EXCP-06, REFR-01, REFR-03, LOG-01, LOG-02 | 10 |
| Phase 2 | CIRCB-01, CIRCB-02, CIRCB-03, CIRCB-04, CIRCB-05, CIRCB-06, CIRCB-07, LOG-03 | 8 |
| Phase 3 | TEST-01, TEST-02, TEST-03, TEST-04, TEST-05, TEST-06, TEST-07, REFR-02 | 8 |
| Phase 4 | Integration (implicit) | 3 |
| Phase 5 | Integration (implicit) | 3 |

### Coverage Summary

- **Total v1 requirements:** 29
- **Mapped to phases:** 29
- **Unmapped:** 0 ✓
- **Coverage:** 100%

---

## Dependencies

### Phase Dependencies

```
Phase 1: [no dependencies]
    ↓
Phase 2: [Phase 1 complete]
    ↓
Phase 3: [Phase 1 complete] (can run in parallel with Phase 2)
    ↓
Phase 4: [Phase 2 AND Phase 3 complete]
    ↓
Phase 5: [Phase 4 complete]
```

### Parallelization Notes

- **Phase 2 and Phase 3** can execute in parallel since they depend only on Phase 1
- **Phase 4** requires both Phase 2 (circuit breaker) and Phase 3 (tests) to be complete
- **Phase 5** depends on Phase 4 integration completion

---

## Notes

### Implementation Order Within Phases

Within each phase, requirements should be implemented in the order listed for optimal dependency resolution. For example, in Phase 1, EXCP-01 must precede EXCP-02, which must precede EXCP-03, etc.

### Research Flags

From research/SUMMARY.md, phases identified as needing deeper research during planning:

- **Phase 4 (Business Logic Integration):** Complex interactions between exception handling, circuit breakers, and guardrail validation
- **Phase 5 (API and Integration):** FastAPI exception handler design and HTTP status code mapping

### Out of Scope (v2+)

These features are deferred to future releases:
- LangChain 2.x migration
- Property-based testing (Hypothesis)
- Mutation testing
- Adaptive circuit breaker thresholds
- Distributed circuit coordination

---

*Roadmap created: 2026-02-12*
*Ready for planning: yes*
