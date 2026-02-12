# Common Pitfalls: Exception Handling, Testing, and Circuit Breakers in Legacy Python

This document catalogs common pitfalls and mistakes encountered when improving exception handling, adding tests, and implementing circuit breakers in legacy Python codebases.

## 1. Exception Handling Pitfalls

### 1.1 Over-Narrow Exception Catching
**Warning Signs:** Code catching only specific exceptions when multiple types can be raised.
**Prevention:** Use static analysis to identify all exception types, use exception groups.

### 1.2 Silent Failure Anti-Pattern
**Warning Signs:** `except: pass` or `except Exception: pass` patterns.
**Prevention:** Always log exceptions, implement recovery or propagation.

### 1.3 Losing Exception Context
**Warning Signs:** Stack traces that stop at helper function boundaries.
**Prevention:** Use `raise ... from original_exception` for chaining.

## 2. Testing Pitfalls

### 2.1 Testing Wrong Things (Over-Mocking)
**Warning Signs:** 100% coverage but low confidence in code correctness.
**Prevention:** Test behavior, not implementation. Avoid excessive mocking.

### 2.2 Brittle Tests from Over-Specification
**Warning Signs:** Tests fail after any refactoring.
**Prevention:** Test public interfaces, not internal details.

### 2.3 Missing Edge Case Coverage
**Warning Signs:** Tests only covering happy path scenarios.
**Prevention:** Use property-based testing (Hypothesis).

### 2.4 Test Interdependencies
**Warning Signs:** Tests pass individually but fail together.
**Prevention:** Use fixtures for isolation, run in random order.

## 3. Circuit Breaker Mistakes

### 3.1 Incorrect Failure Thresholds
**Warning Signs:** Circuit trips during normal traffic or never trips.
**Prevention:** Start conservative, tune based on production telemetry.

### 3.2 Half-Open State Issues
**Warning Signs:** Requests intermittently failing after recovery.
**Prevention:** Proper half-open state with single probe and timeout.

### 3.3 Thread Safety Issues
**Warning Signs:** Intermittent state inconsistencies across threads.
**Prevention:** Use locks or atomic operations for state changes.

## Summary
| Category | Primary Phase | Key Activities |
|----------|---------------|----------------|
| Exception Handling | Analysis/Mapping | Static analysis, dependency review |
| Testing | Test Design | Behavior-focused testing |
| Circuit Breakers | Implementation | State machine correctness |

See detailed research in ARCHITECTURE.md and FEATURES.md for implementation guidance.
