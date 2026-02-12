# Engineering Health: Exception Handling, Testing & Circuit Breakers Research Summary

**Project:** Engineering Health Infrastructure  
**Domain:** Python Exception Handling, Test Infrastructure & Resilience Patterns  
**Researched:** 2026-02-12  
**Confidence:** HIGH

## Executive Summary

This research establishes a comprehensive foundation for implementing engineering health infrastructure in Python 3.13+ codebases, focusing on three interconnected pillars: exception handling, test infrastructure, and circuit breaker patterns. The research synthesizes industry best practices to create a production-grade system that prevents cascade failures, enables confident refactoring, and provides observable system behavior.

**The recommended approach builds these three pillars as decoupled but integrated components.** Exception handling defines structured error types with hierarchical taxonomy and context preservation; test infrastructure provides comprehensive coverage measurement with isolation and mocking strategies; circuit breakers implement state machines that prevent cascade failures while enabling graceful degradation. Together, they form a resilience architecture where errors are well-defined and recoverable, tests verify correctness without brittleness, and failures are contained without bringing down entire systems.

**Key risks and mitigations center on implementation ordering and testing strategies.** The critical risk is building circuit breakers before foundational exception hierarchies exist, which creates fragile error handling. Another significant risk is over-mocked tests that achieve high coverage without verifying actual behavior. The mitigation is strict adherence to the five-phase build order (Foundation → Core Abstractions → Business Logic → Integration → API), and behavior-focused testing that tests public interfaces rather than implementation details.

## Key Findings

### Recommended Stack

**Summary from STACK.md:** The research recommends a Python-native stack leveraging existing project dependencies (Pydantic) while adding specialized libraries for testing and resilience. No external async-aware circuit breaker library meets requirements, so a custom implementation is recommended. Structured logging is essential for observability, and async testing infrastructure must be configured for Python 3.13's improved asyncio features.

**Core technologies:**

- **Pydantic Custom Exceptions (existing)**: The project already depends on `pydantic-settings>=2.12.0`, which provides the foundation for structured error handling. Custom exception classes leverage Pydantic's `ValidationError` patterns while building a native Python exception hierarchy with enum-based error codes, context dictionaries, and cause chaining for proper traceback preservation.

- **pytest + pytest-asyncio (existing)**: The core testing framework with `asyncio_mode="auto"` enabled eliminates repetitive `@pytest.mark.asyncio` decorators and properly handles async fixture cleanup. This is the foundation for all test infrastructure.

- **pytest-mock + respx (required)**: `pytest-mock>=3.14.0` provides consistent `.mock` fixtures across all tests, while `respx>=0.21.0` enables HTTP request mocking for async operations. These libraries are essential for isolated, fast unit tests.

- **structlog (required)**: `structlog>=24.2.0` provides best-in-class structured logging with JSON output support. All logs are structured for easy parsing by observability platforms, with context enrichment and consistent formatting.

- **Custom Async Circuit Breaker (required implementation)**: No widely-adopted async-aware circuit breaker library exists that meets Python 3.13's asyncio capabilities. A custom implementation is recommended with clear state machine (CLOSED → OPEN → HALF_OPEN), statistics tracking, callbacks for logging/alerting, and thread-safe concurrent access.

### Expected Features

**Summary from FEATURES.md:** Engineering health features fall into table stakes (essential), differentiators (competitive advantage), and anti-features (to avoid). Table stakes represent minimum viable production-grade infrastructure across all three pillars. Differentiators provide significant engineering advantages but require additional investment. Anti-features are patterns that actively undermine system reliability and must be avoided.

**Must have (table stakes):**

- **Hierarchical Exception Taxonomy**: Custom exception classes inheriting from `BaseGraphRLMError` with enum-based error codes organized by domain (CORE_1xxx, GRAPH_2xxx, SKILL_3xxx, EXTERNAL_4xxx). Enables catching at different levels (`except GraphError` catches all graph-related issues).

- **Context-Aware Exception Logging**: Every exception captures request IDs, session metadata, operation context, and state variables. Python 3.13's structured logging produces JSON-formatted logs for observability platform ingestion.

- **Graceful Degradation Patterns**: Fallback behaviors for external service failures, cached response serving when data sources are unavailable, and circuit breaker integration to prevent cascade failures.

- **Pytest Framework with Modern Features**: Async mode enabled, parameterization for combinatorial testing, fixtures for dependency injection, and CI/CD integration with coverage thresholds (80%+ for new code).

- **Circuit Breaker State Machine**: Three-state machine (CLOSED, OPEN, HALF_OPEN) with configurable failure thresholds, context integration for async code, and graceful fallback when circuits open.

**Should have (competitive):**

- **Automatic Exception Translation Layer**: Converts domain exceptions to API-appropriate errors automatically, preserving debugging info in development while providing user-friendly messages in production.

- **Property-Based Testing with Hypothesis**: Generates hundreds of edge case inputs automatically, finding bugs that manual test writing would miss. Particularly valuable for pure functions and data transformation pipelines.

- **Mutation Testing**: Verifies tests actually catch bugs by introducing mutations (changed operators, removed conditions) and ensuring tests fail. Drives test improvement and identifies code that appears tested but isn't.

- **Circuit Breaker Metrics and Observability**: Integration with alerting systems based on exception thresholds, dashboards showing exception trends, and circuit state distribution metrics.

**Defer (v2+):**

- **Adaptive Circuit Breaking**: Machine learning models that predict downstream service health and adjust circuit sensitivity. Complex and requires significant production telemetry to tune effectively.

- **Distributed Circuit Coordination**: Cross-service circuit breaker state consistency using consensus protocols or distributed caches. Essential for complex microservices but overkill for initial implementation.

- **Cross-Language Exception Compatibility**: Polyglot translation layers for Go, Java, and other languages. Defer until actual multi-language services exist.

### Architecture Approach

**Summary from ARCHITECTURE.md:** The architecture defines clear component boundaries with decoupled responsibilities: Error Handling defines and propagates exceptions, Logging captures diagnostic information without business logic coupling, and Circuit Breakers monitor failure patterns and block operations when thresholds are exceeded. Data flows through async systems with correlation ID propagation across all layers, from API through Agent to Service to Infrastructure layers.

**Major components:**

1. **Error Handling Component** (`src/core/exceptions/`): Defines exception hierarchy with base classes (BaseGraphRLMError) and specific subclasses (CoreError, GraphError, SkillExecutionError, ExternalServiceError). Carries contextual information including error codes, correlation IDs, and metadata for recovery suggestions.

2. **Logging Component** (`src/core/logging/`): Structured logging with context enrichment via Python's `contextvars`. Handles level-based filtering, output routing, and integration with the exception system for unified observability.

3. **Circuit Breaker Component** (`src/core/resilience/`): State machine implementation with CLOSED/OPEN/HALF_OPEN transitions. Monitors failure rates, enforces thresholds, manages recovery windows, and provides fallback mechanisms. Both async and sync implementations for different operation types.

4. **Test Infrastructure** (`tests/`): Organized by functionality with conftest.py for global fixtures, legacy/ for legacy file tests, integration/ for cross-component tests, and mocking/ for reusable mock libraries.

### Critical Pitfalls

1. **Over-Narrow Exception Catching**: Catching only specific exceptions when multiple types can be raised masks errors and prevents proper diagnosis. Prevention: Use static analysis to identify all exception types and leverage Python 3.13's exception groups for multiple exception handling.

2. **Silent Failure Anti-Pattern**: `except: pass` or bare `except Exception: pass` patterns silently swallow exceptions without logging, notification, or remediation. Prevention: Every caught exception must either be logged, trigger an alert, or be part of an intentional graceful degradation strategy documented in code review.

3. **Over-Mocking Tests**: 100% coverage with excessive mocking verifies nothing about actual system behavior. Prevention: Test behavior through public interfaces, not implementation details. Use minimal mocking to isolate external dependencies only.

4. **Incorrect Failure Thresholds**: Circuit breakers that trip during normal traffic or never trip indicate poorly tuned thresholds. Prevention: Start conservative (5 failures, 60-second timeout) and tune based on production telemetry. Avoid one-size-fits-all configuration.

5. **Thread Safety Issues in Circuit Breakers**: Intermittent state inconsistencies across threads in async environments. Prevention: Use asyncio locks for state changes in async code, atomic operations where possible. Custom circuit breaker implementation must be explicitly designed for concurrent access.

## Implications for Roadmap

Based on research, the engineering health implementation should proceed in five phases that respect build order dependencies and minimize risk. Each phase delivers foundational capabilities that subsequent phases depend upon.

### Phase 1: Foundation
**Rationale:** Establishes the base infrastructure that all other components depend upon. Configuration, error definitions, and logging must exist before any business logic can handle failures gracefully.

**Delivers:**
- Pydantic-based exception hierarchy with enum error codes (`src/core/exceptions/base.py`)
- Structured logging configuration with structlog (`src/core/logging/config.py`)
- Centralized error handler for production monitoring

**Addresses:** Table stakes exception handling features including hierarchical taxonomy, context-aware logging, and centralized error handling.

**Avoids:** Silent failure anti-pattern by ensuring all exceptions are logged and tracked.

### Phase 2: Core Abstractions
**Rationale:** Builds the circuit breaker pattern that protects downstream operations. Requires error handling and logging to be in place for proper state transition logging and error recording.

**Delivers:**
- Async-aware circuit breaker with state machine (`src/core/resilience/circuit_breaker.py`)
- Sync circuit breaker for non-async operations
- Circuit breaker metrics and callbacks for observability
- Context propagation utilities for correlation ID tracking

**Implements:** Circuit breaker component from architecture with CLOSED/OPEN/HALF_OPEN states, failure threshold configuration, and graceful fallback mechanisms.

**Uses:** structlog for circuit state transition logging, custom exceptions for circuit open errors.

### Phase 3: Test Infrastructure
**Rationale:** Establishes comprehensive testing capability before building business logic. This ensures all subsequent code has test coverage from the start, preventing technical debt accumulation.

**Delivers:**
- pytest configuration with asyncio auto mode
- Mock library for legacy file dependencies (`tests/mocking/mocks.py`)
- Test fixtures for async operations and context isolation
- Coverage measurement with per-module tracking

**Uses:** pytest-asyncio, pytest-mock, respx for HTTP mocking, httpx for async HTTP testing.

**Avoids:** Over-mocking by testing behavior through public interfaces and using minimal mocking for external dependencies only.

### Phase 4: Business Logic Integration
**Rationale:** Implements agent core, service layer, and validation logic using the established foundation. All business logic automatically benefits from exception hierarchy, circuit breaker protection, and test infrastructure.

**Delivers:**
- Agent core with integrated circuit breaker protection
- Service layer with graceful degradation patterns
- Guardrail validation with structured error responses

**Implements:** Error handling component with business-specific exceptions, circuit guard decorator for async operations, and exception handlers for API translation.

### Phase 5: API and Integration
**Rationale:** Finalizes the external interfaces and completes integration with external services (LLM, graph database, MCP tools). All integration code is protected by circuit breakers and tested with the established infrastructure.

**Delivers:**
- FastAPI exception handlers for HTTP error responses
- WebSocket streaming with context propagation
- CLI interface with proper error reporting
- Integration tests for all external service calls

### Phase Ordering Rationale

- **Dependency-driven ordering**: Configuration must precede logging, which must precede circuit breakers. Each phase builds on the previous foundation without circular dependencies.
- **Risk minimization**: Starting with exception handling and logging ensures all subsequent errors are visible and traceable. Circuit breakers protect already-visible errors. Tests verify the entire stack.
- **Legacy file handling**: The test infrastructure phase specifically addresses legacy files through analysis, mock library creation, and incremental coverage strategies.
- **Pitfall avoidance**: The ordered approach prevents the most common failure mode—building circuit breakers on fragile error handling—by establishing robust exceptions first.

### Research Flags

Phases likely needing deeper research during planning:

- **Phase 4 (Business Logic Integration)**: Complex interactions between exception handling, circuit breakers, and guardrail validation. May need additional research on specific error recovery patterns for each business domain.
- **Phase 5 (API and Integration)**: FastAPI exception handler design and HTTP status code mapping requires careful API design research to ensure proper error responses.

Phases with standard patterns (skip research-phase):

- **Phase 1 (Foundation)**: Well-documented patterns from Python and Pydantic documentation. No additional research needed beyond what's in STACK.md.
- **Phase 2 (Core Abstractions)**: Standard circuit breaker pattern implementation with custom async adaptation. Architecture is fully specified in ARCHITECTURE.md.
- **Phase 3 (Test Infrastructure)**: Pytest and mocking patterns are standard. Legacy file strategy is fully specified with automated analysis tools.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Leverages existing Pydantic dependency, adds well-established testing libraries, custom circuit breaker design is fully specified |
| Features | HIGH | Table stakes features are standard engineering practices; differentiators are optional enhancements with clear v2 deferral criteria |
| Architecture | HIGH | Component boundaries clearly defined, async data flow thoroughly specified, build order dependencies explicitly documented |
| Pitfalls | HIGH | All major pitfalls identified with specific prevention strategies; patterns validated across multiple sources |

**Overall confidence:** HIGH

### Gaps to Address

- **Circuit breaker configuration tuning**: Initial thresholds (5 failures, 60-second timeout, 2 success threshold) are conservative recommendations that should be validated against production telemetry during implementation.

- **Legacy file coverage targets**: The incremental coverage strategy prioritizes files but specific coverage percentages should be validated during Phase 3 implementation based on actual legacy file complexity.

## Sources

### Primary (HIGH confidence)
- Python 3.13 official documentation — asyncio improvements, exception groups, type system enhancements
- Pydantic documentation — ValidationError patterns, settings management
- pytest documentation — async fixtures, configuration options, plugin ecosystem

### Secondary (MEDIUM confidence)
- Circuit breaker academic papers and pattern descriptions — state machine design, failure threshold calculation
- structlog documentation — structured logging best practices, JSON output formatting
- pytest-asyncio documentation — async test patterns, fixture scoping

### Tertiary (LOW confidence)
- Community circuit breaker implementations — async adaptation patterns, need validation in Python 3.13
- Legacy testing strategies — context-dependent, should be validated against actual legacy files during implementation

---
*Research completed: 2026-02-12*
*Ready for roadmap: yes*