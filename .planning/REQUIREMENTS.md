# Requirements: Graph-RLM Engineering Health

**Defined:** 2026-02-12
**Core Value:** Fix foundational reliability issues before adding capabilities

## v1 Requirements

Requirements for initial engineering health release. All map to roadmap phases.

### Exception Handling

- [ ] **EXCP-01**: Create base exception class `BaseGraphRLMError` with context preservation (correlation_id, timestamp, cause chaining)
- [ ] **EXCP-02**: Define ErrorCode enum with hierarchical categories (CORE_*, GRAPH_*, SKILL_*, EXTERNAL_*)
- [ ] **EXCP-03**: Implement specific exception types: `GraphError`, `SkillExecutionError`, `ExternalServiceError`, `ValidationError`
- [ ] **EXCP-04**: Replace 141+ broad `except Exception` handlers with specific exception types
- [ ] **EXCP-05**: Add structured logging to all exception handlers with full context
- [ ] **EXCP-06**: Implement exception-to-dict serialization for API error responses

### Test Infrastructure

- [ ] **TEST-01**: Configure pytest with `asyncio_mode = auto` and proper test paths
- [ ] **TEST-02**: Create mock registry (`tests/mocking/mocks.py`) for FalkorDB, LLM service, external APIs
- [ ] **TEST-03**: Create base test fixtures for async setup/teardown
- [ ] **TEST-04**: Implement MockRegistry class with reset capability
- [ ] **TEST-05**: Add first unit tests for `src/core/config.py` (100% coverage goal)
- [ ] **TEST-06**: Add first unit tests for `src/core/exceptions/base.py` (100% coverage goal)
- [ ] **TEST-07**: Configure pytest-cov for incremental coverage tracking

### Circuit Breaker

- [ ] **CIRCB-01**: Implement CircuitState enum (CLOSED, OPEN, HALF_OPEN)
- [ ] **CIRCB-02**: Create CircuitBreakerConfig dataclass with failure_threshold, timeout, success_threshold
- [ ] **CIRCB-03**: Implement async-aware CircuitBreaker class with state machine
- [ ] **CIRCB-04**: Add CircuitOpenError exception for open state rejection
- [ ] **CIRCB-05**: Integrate circuit breaker with LLM service calls
- [ ] **CIRCB-06**: Integrate circuit breaker with MCP server connections
- [ ] **CIRCB-07**: Implement metrics/observability hooks for circuit state transitions

### Refactoring

- [ ] **REFR-01**: Remove duplicate LLM_PROVIDER config definition in Settings class
- [ ] **REFR-02**: Create isolated test module for `src/core/agent.py` with legacy mocking
- [ ] **REFR-03**: Add type hints to exception base class and error code enum

### Logging & Observability

- [ ] **LOG-01**: Configure structlog for JSON-structured logging
- [ ] **LOG-02**: Implement exception context enrichment for all logs
- [ ] **LOG-03**: Add correlation_id propagation through async calls

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Testing

- **TEST-08**: Property-based testing with Hypothesis for edge cases
- **TEST-09**: Mutation testing (mutmut) to verify test quality
- **TEST-10**: Contract testing for API boundaries
- **TEST-11**: Snapshot testing for complex outputs

### Advanced Circuit Breaker

- **CIRCB-08**: Adaptive thresholds based on production telemetry
- **CIRCB-09**: Distributed coordination (Redis-backed state sharing)
- **CIRCB-10**: Hierarchical circuit breakers for nested dependencies
- **CIRCB-11**: Visualization/debugging endpoints for circuit state

### Additional Coverage

- **TEST-12**: Agent core tests (80% coverage of agent.py)
- **TEST-13**: Integration tests for skill execution lifecycle
- **TEST-14**: Integration tests for axiom verification chain

### Exception Enhancements

- **EXCP-07**: Exception group handling (Python 3.13+) for batch operations
- **EXCP-08**: Recovery hint generation based on error type
- **EXCP-09**: Error rate tracking and alerting
- **EXCP-10**: Automatic retry logic integration with backoff

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Large file refactoring (agent.py, dream.py split) | Defer to post-health phase; focus on testability first |
| Security hardening (.env validation, secrets management) | Home lab use case, user-managed keys |
| Rate limiting for LLM APIs | Defer to post-health phase |
| Graph cleanup/TTL policies | Defer to post-health phase |
| Redis session store | Defer to post-health phase |
| LangChain 2.x migration | Research needed, may require API changes |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| EXCP-01 | Phase 1: Foundation | Pending |
| EXCP-02 | Phase 1: Foundation | Pending |
| EXCP-03 | Phase 1: Foundation | Pending |
| EXCP-04 | Phase 1: Foundation | Pending |
| EXCP-05 | Phase 1: Foundation | Pending |
| EXCP-06 | Phase 1: Foundation | Pending |
| REFR-01 | Phase 1: Foundation | Pending |
| REFR-03 | Phase 1: Foundation | Pending |
| LOG-01 | Phase 1: Foundation | Pending |
| LOG-02 | Phase 1: Foundation | Pending |
| CIRCB-01 | Phase 2: Core Abstractions | Pending |
| CIRCB-02 | Phase 2: Core Abstractions | Pending |
| CIRCB-03 | Phase 2: Core Abstractions | Pending |
| CIRCB-04 | Phase 2: Core Abstractions | Pending |
| CIRCB-05 | Phase 2: Core Abstractions | Pending |
| CIRCB-06 | Phase 2: Core Abstractions | Pending |
| CIRCB-07 | Phase 2: Core Abstractions | Pending |
| LOG-03 | Phase 2: Core Abstractions | Pending |
| TEST-01 | Phase 3: Test Infrastructure | Pending |
| TEST-02 | Phase 3: Test Infrastructure | Pending |
| TEST-03 | Phase 3: Test Infrastructure | Pending |
| TEST-04 | Phase 3: Test Infrastructure | Pending |
| TEST-05 | Phase 3: Test Infrastructure | Pending |
| TEST-06 | Phase 3: Test Infrastructure | Pending |
| TEST-07 | Phase 3: Test Infrastructure | Pending |
| REFR-02 | Phase 3: Test Infrastructure | Pending |

**Phase Summary:**
- **Phase 1 (Foundation):** 10 requirements (EXCP-01~06, REFR-01, REFR-03, LOG-01~02)
- **Phase 2 (Core Abstractions):** 8 requirements (CIRCB-01~07, LOG-03)
- **Phase 3 (Test Infrastructure):** 8 requirements (TEST-01~07, REFR-02)
- **Phase 4 (Business Logic Integration):** Integration requirements (implicit, not explicit v1)
- **Phase 5 (API and Integration):** Integration requirements (implicit, not explicit v1)

**Coverage:**
- v1 requirements: 26 total (explicit, mappable)
- Mapped to phases: 26
- Unmapped: 0 ✓
- Integration requirements: 3 (handled in Phases 4-5)

---

*Requirements defined: 2026-02-12*
*Last updated: 2026-02-12 after research synthesis*
