# Phase 2: Core Abstractions - Research

**Researched:** 2026-02-12
**Domain:** Async Circuit Breaker Pattern Implementation in Python 3.13
**Confidence:** MEDIUM-HIGH

## Summary

Phase 2 requires implementing a custom async-aware circuit breaker pattern for Graph-RLM's resilience against external service failures. The circuit breaker must integrate with:
- LLM service calls (async operations, potential streaming)
- MCP server connections
- Structured logging with correlation ID propagation
- Metrics/observability hooks for state transitions

**Key finding:** The decision to build a custom circuit breaker is appropriate. Most existing Python circuit breaker libraries (pybreaker, circuitbreaker, async-circuitbreaker) have limitations with Python 3.13's async features, lack proper async context manager support, or don't provide the observability hooks needed for Graph-RLM's structured logging requirements.

**Primary recommendation:** Implement a custom `CircuitBreaker` class with `CircuitState` enum, `CircuitBreakerConfig` dataclass, and integration patterns that leverage Phase 1's exception hierarchy and structlog configuration.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python 3.13 | 3.13+ | Base language | Native async/await, improved exception groups |
| asyncio | built-in | Async runtime | Event loop, tasks, futures |
| structlog | 24.2+ | Structured logging | Configured in Phase 1, required for observability |

### Integration Points
| Component | Purpose | Interface |
|-----------|---------|-----------|
| LLM Service | External AI API calls | async functions, potential streaming |
| MCP Server | Model Context Protocol | Connection management, async operations |
| Exception Hierarchy | Error handling | BaseGraphRLMError subclasses |
| Structured Logging | Observability | structlog integration with correlation_id |

### No External Circuit Breaker Libraries
Building custom implementation - see "Don't Hand-Roll" section for rationale.

## Architecture Patterns

### Recommended Project Structure

```
src/core/
├── circuit.py           # CircuitState enum, CircuitBreakerConfig, CircuitBreaker class
├── exceptions/
│   └── circuit.py       # CircuitOpenError (extends BaseGraphRLMError)
└── logging/
    └── correlation.py   # Correlation ID propagation utilities

tests/
├── unit/
│   └── test_circuit.py  # 100% coverage on CircuitBreaker
└── conftest.py          # pytest fixtures for circuit breaker tests
```

### Pattern 1: Circuit State Machine
**What:** Three-state finite state machine for circuit breaker
**When to use:** Core circuit breaker implementation

```python
from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation, counting failures
    OPEN = auto()        # Rejecting calls immediately
    HALF_OPEN = auto()   # Testing recovery, limited calls

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5          # Failures before opening
    timeout_seconds: float = 60.0       # Time in OPEN before HALF_OPEN
    success_threshold: int = 3          # Successes in HALF_OPEN to close
```

### Pattern 2: Async-Aware Circuit Breaker
**What:** Context manager that handles async function wrapping and state transitions
**When to use:** Protecting async LLM and MCP service calls

```python
class CircuitBreaker:
    """Async-aware circuit breaker with state machine."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: datetime | None = None
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> CircuitState:
        """Get current state, transitioning OPEN -> HALF_OPEN if timeout expired."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time:
                elapsed = datetime.now() - self._last_failure_time
                if elapsed >= timedelta(seconds=self.config.timeout_seconds):
                    self._state = CircuitState.HALF_OPEN
        return self._state
    
    async def call_async(self, func, *args, **kwargs):
        """Execute async function through circuit breaker."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is open",
                    correlation_id=get_correlation_id()
                )
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure(e)
            raise
    
    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("circuit_closed", circuit=self.name)
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success in CLOSED state
                self._failure_count = 0
    
    async def _on_failure(self, error: Exception):
        """Handle failed call."""
        async with self._lock:
            self._last_failure_time = datetime.now()
            if self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning("circuit_opened", circuit=self.name, error=str(error))
            elif self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._success_count = 0
                logger.warning("circuit_reopened", circuit=self.name, error=str(error))
```

### Pattern 3: Exception with Context
**What:** CircuitOpenError extending BaseGraphRLMError
**When to use:** When circuit is open and calls are rejected

```python
class CircuitOpenError(BaseGraphRLMError):
    """Raised when circuit breaker is open and calls are rejected."""
    
    def __init__(self, message: str, correlation_id: str | None = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.CIRCUIT_OPEN,
            correlation_id=correlation_id
        )
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Async circuit breaker | Custom from scratch | Build custom (see rationale) | Existing libraries lack Python 3.13 async support |
| LLM call integration | Direct calls without protection | CircuitBreaker.call_async() wrapper | Consistent failure handling |
| MCP connection pooling | Custom implementation | Circuit breaker pattern on connection | Simplified, testable |
| State persistence | Distributed state store | In-memory (local circuit breaker) | Graph-RLM is single-instance |

### Why Build Custom (Rationale)

**pybreaker** (https://github.com/danielfm/pybreaker):
- Last updated 2020, no Python 3.13 support
- Synchronous only, no async/await support
- No structured logging hooks

**circuitbreaker** (https://github.com/fabiorogeriosilva/circuitbreaker):
- Python 3.9+ but sync-focused
- Limited async integration points
- No correlation ID propagation

**async-circuitbreaker** (various forks):
- Multiple unmaintained implementations
- Incompatible with Python 3.13's new async features
- Missing observability hooks for structlog

**Custom implementation advantages:**
1. Full async/await integration with Python 3.13
2. Structured logging via structlog (already configured in Phase 1)
3. Correlation ID propagation through async context
4. Integration with Phase 1 exception hierarchy (CircuitOpenError extends BaseGraphRLMError)
5. Metrics hooks for observability
6. Testable patterns with MockRegistry from Phase 3

## Common Pitfalls

### Pitfall 1: Lock Contention in High-Concurrency Scenarios
**What goes wrong:** Global lock on every async call creates bottlenecks
**Why it happens:** Circuit breaker state changes need synchronization, but locking every call is expensive
**How to avoid:** Use fine-grained locking - only lock state transitions, not the actual function call
**Warning signs:** Async timeouts, slow LLM responses, lock contention errors

### Pitfall 2: State Transitions Without Logging
**What goes wrong:** Circuit opens/closes silently, no observability
**Why it happens:** Missing structured logging hooks in state transition methods
**How to avoid:** Always log state transitions with structlog, include correlation_id
**Warning signs:** No circuit state logs, inability to diagnose cascading failures

### Pitfall 3: Ignoring Exception Types in Failure Count
**What goes wrong:** Counting all exceptions equally, including transient network errors
**Why it happens:** Simple exception catch without distinguishing error types
**How to avoid:** Implement failure type filtering - don't count retry-able errors as circuit-opening failures
**Warning signs:** Circuit opens on temporary network blips, over-aggressive protection

### Pitfall 4: Missing Timeout Coordination
**What goes wrong:** Circuit breaker timeout doesn't coordinate with LLM service timeouts
**Why it happens:** Independent timeout configurations without alignment
**How to avoid:** Coordinate circuit breaker timeout with LLM service timeout + buffer
**Warning signs:** LLM calls timing out before circuit opens, confusing error sequences

## Code Examples

### Protected LLM Service Call

```python
from src.core.circuit import CircuitBreaker, CircuitBreakerConfig
from src.core.exceptions import CircuitOpenError
from src.core.logging import get_correlation_id, logger

# Create circuit breaker for LLM service
llm_circuit = CircuitBreaker(
    name="llm_service",
    config=CircuitBreakerConfig(
        failure_threshold=3,
        timeout_seconds=30.0,
        success_threshold=2
    )
)

# Protected async call
async def protected_llm_query(prompt: str, **kwargs):
    correlation_id = get_correlation_id()
    logger.info("llm_query_start", prompt=prompt[:100], correlation_id=correlation_id)
    
    try:
        result = await llm_circuit.call_async(
            llm_service.query,
            prompt,
            **kwargs
        )
        logger.info("llm_query_success", correlation_id=correlation_id)
        return result
    except CircuitOpenError:
        logger.warning("llm_query_circuit_open", correlation_id=correlation_id)
        raise
```

### MCP Server Connection with Circuit Breaker

```python
from src.core.circuit import CircuitBreaker, CircuitBreakerConfig

mcp_circuit = CircuitBreaker(
    name="mcp_server",
    config=CircuitBreakerConfig(
        failure_threshold=5,
        timeout_seconds=60.0,
        success_threshold=3
    )
)

async def safe_mcp_call(func, *args, **kwargs):
    """Wrapper for any MCP server call."""
    return await mcp_circuit.call_async(func, *args, **kwargs)
```

### Correlation ID Propagation

```python
import contextvars
from structlog import get_logger

correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)

def get_correlation_id() -> str:
    """Get correlation ID from context or generate new one."""
    cid = correlation_id_var.get()
    if not cid:
        import uuid
        cid = str(uuid.uuid4())[:8]
        correlation_id_var.set(cid)
    return cid

async def with_correlation_id(func, *args, **kwargs):
    """Execute function with correlation ID propagation."""
    token = correlation_id_var.set(get_correlation_id())
    try:
        result = await func(*args, **kwargs)
        return result
    finally:
        correlation_id_var.reset(token)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| No circuit breaker | Custom async-aware implementation | Phase 2 | Protection against cascade failures |
| Generic exception handlers | Specific exception types + CircuitOpenError | Phase 1 + 2 | Better error diagnosis |
| Unstructured logs | structlog with correlation_id | Phase 1 + 2 | Observable failure patterns |
| Synchronous only | Async-first with Python 3.13 | Phase 2 | Better LLM/MCP integration |

**Deprecated/outdated:**
- Synchronous circuit breaker libraries - don't use for async services
- Global exception catching - Phase 1 exception hierarchy handles this
- Manual correlation ID passing - contextvars handles propagation

## Open Questions

1. **MCP Server Health Checks**
   - What we know: MCP servers need connection management, failures should trigger circuit
   - What's unclear: Should we implement explicit health check pings, or rely on actual call failures?
   - Recommendation: Start with actual call failures, add health checks if false-positive circuits occur

2. **Streaming Response Handling**
   - What we know: LLM services may return streaming responses
   - What's unclear: How does circuit breaker handle partial stream completion vs. failure mid-stream?
   - Recommendation: Track success/failure at stream completion level, not per-chunk

3. **Circuit Breaker Granularity**
   - What we know: Graph-RLM has multiple external services (LLM, MCP, potentially others)
   - What's unclear: Single circuit breaker for all services, or per-service circuits?
   - Recommendation: Per-service circuits (LLM, MCP) - independent failure domains

## Sources

### Primary (HIGH confidence)
- Python asyncio documentation - https://docs.python.org/3/library/asyncio.html
- structlog documentation - https://www.structlog.org/
- Circuit breaker pattern (Martin Fowler) - https://martinfowler.com/bliki/CircuitBreaker.html

### Secondary (MEDIUM confidence)
- Various Python circuit breaker library GitHub repos (analyzed for gaps)
- Python 3.13 async features - https://docs.python.org/3/whatsnew/3.13.html

### Tertiary (LOW confidence)
- Community async patterns - marked for validation during implementation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Python async patterns well-established
- Architecture: HIGH - State machine pattern is standard
- Don't-hand-roll rationale: HIGH - Analyzed existing libraries
- Pitfalls: MEDIUM - Based on common patterns, some Python 3.13 specifics need validation

**Research date:** 2026-02-12
**Valid until:** 2026-08-12 (6 months - async patterns stable)

## User Constraints

No CONTEXT.md exists for this phase - no user decisions to constrain research scope.
