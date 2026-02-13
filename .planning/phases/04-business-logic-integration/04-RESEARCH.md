# Phase 4: Business Logic Integration - Research

**Researched:** 2026-02-12
**Domain:** Circuit Breaker Integration, Graceful Degradation, Error Handling Patterns
**Confidence:** HIGH

## Summary

Phase 4 integrates the circuit breaker pattern and exception handling into the Graph-RLM agent core with graceful degradation capabilities. The existing infrastructure from Phases 1-3 provides all necessary building blocks: exception hierarchy (`BaseGraphRLMError`), circuit breaker (`CircuitBreaker`, `llm_circuit`, `mcp_circuit`), and correlation ID propagation. The focus is on integrating these into `agent.main_loop`, implementing structured error logging with full context, and establishing validation patterns for graceful degradation when external services fail.

**Primary recommendation:** Integrate `protected_llm_generate` from `services/circuit.py` into agent.main_loop at the LLM call site (agent.py:820), add circuit state transition logging, and implement fallback behaviors for graceful degradation.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python 3.13 | 3.13+ | Base language | Native async/await, contextvars |
| asyncio | built-in | Async runtime | Already in use throughout codebase |
| structlog | 24.2+ | Structured logging | Configured in Phase 1 |

### Infrastructure (Phases 1-3)
| Component | Source | Purpose |
|-----------|--------|---------|
| `CircuitBreaker` | `src/core/circuit.py` | Circuit protection for external services |
| `llm_circuit` | `src/core/circuit.py:392` | Pre-configured LLM service circuit |
| `mcp_circuit` | `src/core/circuit.py:402` | Pre-configured MCP server circuit |
| `BaseGraphRLMError` | `src/core/exceptions/base.py` | Base exception with context |
| `ValidationError` | `src/core/exceptions/types.py` | Input validation errors |
| `protected_llm_generate` | `src/core/services/circuit.py` | Circuit-breaker-protected LLM calls |

### Integration Points
| Component | Purpose | Integration Method |
|-----------|---------|-------------------|
| `agent.main_loop` | Core agent execution | Import protected LLM functions |
| `LLMService.generate()` | LLM API calls | Wrap with circuit breaker |
| `Exception handlers` | Error capture | Use existing exception hierarchy |
| `Logger` | Structured logging | Leverage existing structlog config |

## Architecture Patterns

### Recommended Integration Structure

```
graph_rlm/backend/src/core/
├── agent.py                    # Main integration target
├── circuit.py                  # CircuitBreaker (Phase 2)
├── services/
│   ├── circuit.py              # Protected LLM functions (Phase 2)
│   └── __init__.py             # Export protected functions
└── exceptions/                 # Exception hierarchy (Phase 1)
```

### Pattern 1: Agent Circuit Breaker Integration
**What:** Wrap LLM service calls in agent.main_loop with circuit breaker protection
**When to use:** Every LLM call site in agent execution flow

```python
# In agent.py (line ~820), replace direct LLM call with:
from .services.circuit import protected_llm_generate, get_correlation_id

async def _call_llm(self, prompt: str, system: str) -> str:
    """Protected LLM call with circuit breaker."""
    correlation_id = get_correlation_id()
    logger.info("llm_call_start", correlation_id=correlation_id)
    
    try:
        result = await protected_llm_generate(
            prompt, 
            system=system,
            correlation_id=correlation_id
        )
        return result
    except CircuitOpenError as e:
        logger.warning("llm_circuit_open", correlation_id=correlation_id)
        return self._handle_llm_circuit_open(e)
```

### Pattern 2: Graceful Degradation with Fallback
**What:** Provide fallback behavior when external services fail
**When to use:** LLM and MCP service calls that should not block agent execution

```python
# In agent.py, add graceful degradation method:
async def _handle_llm_circuit_open(self, error: CircuitOpenError) -> str:
    """Handle LLM circuit open with graceful degradation."""
    logger.error("llm_service_unavailable", 
                 correlation_id=error.correlation_id,
                 circuit=error.circuit_name)
    
    # Emit event for UI feedback
    self.emit_event(
        "error",
        content="AI service temporarily unavailable. Attempting recovery..."
    )
    
    # Try to emit thought about the failure
    return "I notice the AI service is temporarily unavailable. " \
           "Please retry your request."
```

### Pattern 3: Input Validation with ValidationError
**What:** Validate agent inputs using ValidationError pattern
**When to use:** Input validation for prompts, configurations, and external data

```python
from .exceptions import ValidationError, ErrorCode

def validate_agent_input(prompt: str, max_length: int = 100000) -> None:
    """Validate agent input parameters."""
    if not prompt:
        raise ValidationError(
            message="Prompt cannot be empty",
            error_code=ErrorCode.VALIDATION_FIELD_REQUIRED,
            field="prompt",
            constraint="non_empty"
        )
    
    if len(prompt) > max_length:
        raise ValidationError(
            message=f"Prompt exceeds maximum length of {max_length} characters",
            error_code=ErrorCode.VALIDATION_VALUE_OUT_OF_RANGE,
            field="prompt",
            constraint=f"length <= {max_length}",
            actual_length=len(prompt)
        )
```

### Pattern 4: Circuit State Transition Logging
**What:** Log circuit state transitions with full context
**When to use:** Any circuit state change for observability

```python
# In agent.__init__ or circuit initialization:
from .circuit import llm_circuit, get_correlation_id

# Monitor circuit state transitions
def log_circuit_transition(circuit_name: str, old_state: str, new_state: str):
    """Log circuit state transition with correlation ID."""
    correlation_id = get_correlation_id()
    logger.info(
        "circuit_state_transition",
        circuit=circuit_name,
        old_state=old_state,
        new_state=new_state,
        correlation_id=correlation_id
    )

# Monitor circuit metrics periodically
async def monitor_circuit_health():
    """Monitor and log circuit health metrics."""
    metrics = llm_circuit.get_metrics()
    logger.info("circuit_health", **metrics)
```

### Pattern 5: Error Context Propagation
**What:** Ensure all exceptions have full context (correlation_id, operation)
**When to use:** Throughout agent execution for debugging

```python
from .exceptions import GraphError, ErrorCode
from .circuit import get_correlation_id, set_correlation_id

async def execute_graph_operation(operation: str, **params):
    """Execute graph operation with error context."""
    correlation_id = get_correlation_id()
    
    try:
        result = await db.execute_operation(operation, **params)
        return result
    except Exception as e:
        logger.error("graph_operation_failed",
                    operation=operation,
                    correlation_id=correlation_id,
                    error=str(e))
        
        raise GraphError(
            message=f"Graph operation '{operation}' failed",
            error_code=ErrorCode.GRAPH_OPERATION_FAILED,
            correlation_id=correlation_id,
            operation=operation,
            **params
        ) from e
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Circuit breaker | Custom implementation | `CircuitBreaker` from Phase 2 | Already implemented with metrics and logging |
| Protected LLM calls | Custom wrappers | `protected_llm_generate` from Phase 2 | Already handles correlation ID and logging |
| LLM fallback | Custom fallback logic | `protected_llm_with_fallback` | Already handles circuit open gracefully |
| Exception context | Custom error wrapping | `BaseGraphRLMError` subclasses | Already has correlation ID and context support |
| Structured logging | Custom log formatting | structlog (Phase 1) | Already configured with JSON output |
| Validation patterns | Custom validation | `ValidationError` + `ErrorCode` | Already has field, schema, constraint support |

**Key insight:** Phase 4 integration is about wiring existing components together, not implementing new functionality. The infrastructure from Phases 1-3 is production-ready.

## Common Pitfalls

### Pitfall 1: Missing Circuit Breaker Import in Agent
**What goes wrong:** Direct LLM calls bypass circuit protection
**Why it happens:** Using `self.llm.generate()` instead of protected wrapper
**How to avoid:** Import and use `protected_llm_generate` from `services.circuit`
**Warning signs:** No circuit state logs, no graceful degradation on LLM failures

### Pitfall 2: Lost Correlation ID Context
**What goes wrong:** Error logs lack correlation ID for tracing
**Why it happens:** Not using `set_correlation_id` / `get_correlation_id` in agent context
**How to avoid:** Always set correlation ID at entry point and propagate through calls
**Warning signs:** Empty correlation_id in logs, difficult debugging

### Pitfall 3: Overly Broad Exception Handlers
**What goes wrong:** Catching generic `Exception` without specific handling
**Why it happens:** Using bare `except Exception:` without CircuitOpenError handling
**How to avoid:** Specific exception handlers for CircuitOpenError vs other exceptions
**Warning signs:** Circuit breaker opens silently, no fallback behavior

### Pitfall 4: No Graceful Degradation
**What goes wrong:** Agent halts when LLM service is unavailable
**Why it happens:** No fallback behavior when protected_llm_generate fails
**How to avoid:** Implement fallback methods and emit user-facing events
**Warning signs:** User sees raw error messages, agent becomes unresponsive

### Pitfall 5: Missing ValidationError Usage
**What goes wrong:** Input validation errors use generic exceptions
**Why it happens:** Not using `ValidationError` pattern for input validation
**How to avoid:** Use ValidationError for all input validation with proper error codes
**Warning signs:** No VALIDATION_* error codes in logs, unclear error messages

## Code Examples

### Protected LLM Call in Agent Main Loop

```python
# Replace direct call at agent.py:820 with protected version:
from .services.circuit import protected_llm_generate
from .circuit import get_correlation_id

# In agent.main_loop:
async def _call_llm_with_protection(self, current_context: str, system_prompt: str) -> str:
    """Make protected LLM call through circuit breaker."""
    correlation_id = get_correlation_id()
    
    self.emit_event(
        "debug_thought",
        content=f"... Sending request to LLM (Size: {len(current_context)} chars) ..."
    )
    
    try:
        response_text = await protected_llm_generate(
            current_context,
            system=system_prompt,
            stream=False,
            correlation_id=correlation_id
        )
        
        # Check for error response from LLM (not circuit breaker)
        if isinstance(response_text, str) and response_text.startswith("Error:"):
            raise ExternalServiceError(
                message=f"LLM returned error: {response_text}",
                error_code=ErrorCode.EXTERNAL_SERVICE_ERROR,
                correlation_id=correlation_id,
                service="LLM",
                response=response_text
            )
        
        return response_text
        
    except CircuitOpenError as e:
        logger.warning("llm_circuit_open", 
                     correlation_id=e.correlation_id,
                     circuit=e.circuit_name)
        return await self._handle_llm_circuit_open(e)
```

### Graceful Degradation Handler

```python
async def _handle_llm_circuit_open(self, error: CircuitOpenError) -> str:
    """Handle LLM circuit open with graceful degradation."""
    correlation_id = error.correlation_id or get_correlation_id()
    
    # Log with full context
    logger.error("llm_service_degraded",
                correlation_id=correlation_id,
                circuit=error.circuit_name,
                message=error.message)
    
    # Emit event for user feedback
    self.emit_event(
        "error",
        content="The AI service is experiencing high demand. "
                "I'll continue processing with limited capabilities."
    )
    
    # Record error node in graph
    if hasattr(self, 'db') and self.db:
        try:
            thought_id = str(uuid.uuid4())
            self.db.create_thought_node(
                thought_id,
                f"[Circuit Breaker Active] LLM service unavailable. "
                f"Correlation: {correlation_id}. "
                f"Will retry automatically.",
                session_id=self.session_id if hasattr(self, 'session_id') else "default",
                status="degraded"
            )
        except Exception as graph_error:
            logger.warning("Failed to record circuit breaker event in graph: %s", graph_error)
    
    # Return degraded but functional response
    return "AI service temporarily unavailable. " \
           "Please wait a moment and retry your request."
```

### Validation Integration Pattern

```python
from .exceptions import ValidationError, ErrorCode

def validate_user_prompt(prompt: str) -> None:
    """Validate incoming user prompt."""
    if not prompt or not prompt.strip():
        raise ValidationError(
            message="User prompt cannot be empty",
            error_code=ErrorCode.VALIDATION_FIELD_REQUIRED,
            field="user_prompt",
            constraint="non_empty"
        )
    
    max_length = 100000  # 100KB limit
    if len(prompt) > max_length:
        raise ValidationError(
            message=f"Prompt exceeds maximum length of {max_length} characters",
            error_code=ErrorCode.VALIDATION_VALUE_OUT_OF_RANGE,
            field="user_prompt",
            constraint=f"length <= {max_length}",
            actual_length=len(prompt)
        )

# Usage in agent query endpoint:
async def handle_user_query(self, prompt: str) -> str:
    """Handle user query with validation."""
    try:
        validate_user_prompt(prompt)
        return await self.query_sync(prompt)
    except ValidationError as e:
        logger.warning("query_validation_failed",
                      correlation_id=e.correlation_id,
                      error=e.message)
        return f"Invalid query: {e.message}"
```

### Circuit Metrics Monitoring

```python
from .circuit import llm_circuit, mcp_circuit

async def log_circuit_metrics_periodically(interval: int = 60):
    """Periodically log circuit metrics for monitoring."""
    while True:
        try:
            llm_metrics = llm_circuit.get_metrics()
            mcp_metrics = mcp_circuit.get_metrics()
            
            logger.info("circuit_metrics",
                       llm_circuit=llm_metrics,
                       mcp_circuit=mcp_metrics)
            
            await asyncio.sleep(interval)
        except Exception as e:
            logger.error("circuit_metrics_collection_failed: %s", e)
            await asyncio.sleep(interval)

# Start monitoring in agent initialization:
async def start_circuit_monitoring(self):
    """Start circuit health monitoring."""
    asyncio.create_task(log_circuit_metrics_periodically())
    logger.info("circuit_monitoring_started")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Direct LLM calls | Protected through circuit breaker | Phase 4 | Graceful degradation on failures |
| Generic exception handlers | Specific exception types with context | Phase 1 | Better debugging |
| No circuit breaker | CircuitBreaker with metrics | Phase 2 | Cascading failure protection |
| Unstructured logs | structlog with correlation_id | Phase 1 | Observable failure patterns |
| No fallback behavior | Graceful degradation with user feedback | Phase 4 | Continuous operation during outages |
| No input validation | ValidationError patterns | Phase 4 | Clear validation failures |

**Deprecated/outdated:**
- Direct service calls without circuit protection - Phase 4 fixes this
- Generic exception handling - Phase 1 established patterns
- No circuit state observability - Phase 2 provides metrics
- Missing correlation ID context - Phase 2 provides utilities

## Open Questions

1. **MCP Server Circuit Integration**
   - What we know: `mcp_circuit` exists and can protect MCP calls
   - What's unclear: Which MCP operations need circuit protection
   - Recommendation: Add circuit protection to critical MCP calls (file system, network) first

2. **Graph Database Circuit Breaker**
   - What we know: Database operations can fail and cascade
   - What's unclear: Should graph operations have their own circuit breaker?
   - Recommendation: Start without DB circuit breaker; add if needed based on failure patterns

3. **Circuit Breaker Reset Strategy**
   - What we know: Circuits auto-transition from OPEN → HALF_OPEN → CLOSED
   - What's unclear: Manual reset mechanism for production
   - Recommendation: Add admin endpoint to manually reset circuits if needed

4. **Streaming Response Circuit Protection**
   - What we know: `generate()` supports streaming
   - What's unclear: How circuit breaker handles partial stream completion
   - Recommendation: Track success/failure at stream completion level, not per-chunk

## Sources

### Primary (HIGH confidence)
- Phase 1 deliverables: `src/core/exceptions/` - Complete exception hierarchy
- Phase 2 deliverables: `src/core/circuit.py` - Complete circuit breaker implementation
- Phase 2 deliverables: `src/core/services/circuit.py` - Protected LLM service wrappers
- Agent main loop: `src/core/agent.py` - Integration target

### Secondary (MEDIUM confidence)
- Circuit breaker pattern: Phase 2 research established implementation approach
- Validation patterns: Phase 1 research established exception patterns

### Tertiary (LOW confidence)
- Graceful degradation patterns: Industry patterns, needs validation during implementation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All infrastructure exists from Phase 1-3
- Architecture: HIGH - Clear integration patterns documented
- Pitfalls: HIGH - Based on Phase 1-2 implementation and common patterns
- Code examples: HIGH - Based on actual codebase patterns

**Research date:** 2026-02-12
**Valid until:** 2026-08-12 (6 months - infrastructure stable)

## User Constraints

No CONTEXT.md exists for this phase - no user decisions to constrain research scope. All integration decisions are at implementer's discretion following established Phase 1-3 patterns.