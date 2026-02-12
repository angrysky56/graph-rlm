# Exception Handling, Test Infrastructure, and Circuit Breaker Architecture

**Research Date:** 2026-02-12
**Author:** Architecture Research Team

---

## Executive Summary

This document provides a comprehensive analysis of how exception handling frameworks, test infrastructure, and circuit breaker patterns are typically structured in Python codebases. It covers component boundaries, data flow for async systems, build order dependencies, and test infrastructure strategies for legacy codebases.

---

## 1. Component Boundaries: Error Handling vs Logging vs Circuit Breaking

### 1.1 Component Separation Principles

```
+------------------+     +------------------+     +------------------+
|  Error Handling  |     |     Logging      |     |  Circuit Breaker |
+------------------+     +------------------+     +------------------+
| - Exception def  |     | - Log capture    |     | - Failure detect |
| - Error classes  |     | - Formatting     |     | - State machine  |
| - Propagation    |     | - Output sinks   |     | - Fallback paths |
| - Recovery logic  |     | - Level filtering|     | - Recovery timers|
+------------------+     +------------------+     +------------------+
         |                        |                        |
         v                        v                        v
    +---------------------------------------------------+
    |              DECOUPLING LAYER                      |
    |  - Clear interfaces between components            |
    |  - No circular dependencies                        |
    |  - Independent configuration                      |
    +---------------------------------------------------+
```

### 1.2 Error Handling Component

**Purpose:** Define, raise, and catch exceptions with business-context-aware error types.

**Key Responsibilities:**
- Define exception hierarchy (base exception + specific subclasses)
- Carry contextual information (error codes, metadata, correlation IDs)
- Provide error recovery suggestions
- Enable error categorization for monitoring

**Example Structure:**

```python
# exceptions/base.py
class RLMError(Exception):
    """Base exception for all RLM-related errors."""
    error_code: str
    correlation_id: Optional[str] = None
    
    def __init__(self, message: str, correlation_id: Optional[str] = None, **context):
        self.message = message
        self.correlation_id = correlation_id
        self.context = context
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        return f"[{self.error_code}] {self.message}"

# exceptions/agent.py
class AgentError(RLMError):
    """Agent-level errors."""
    error_code = "AGENT_ERR"

# exceptions/guardrails.py
class GuardrailError(AgentError):
    """Guardrail validation failures."""
    error_code = "GUARDRAIL_ERR"
    
    def get_recovery_hint(self) -> str:
        return "Review constraint violations and adjust input parameters."
```

### 1.3 Logging Component

**Purpose:** Capture, format, and route diagnostic information without business logic coupling.

**Key Responsibilities:**
- Structured log event creation
- Level-based filtering (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Output routing (stdout, file, remote collectors)
- Context enrichment (trace IDs, session IDs)
- Performance optimization (sampling, async write)

**Example Structure:**

```python
# logging/logger.py
import logging
import sys
from typing import Optional
from contextvars import ContextVar

correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id")

class StructuredLogger:
    """Structured logging with context awareness."""
    
    def __init__(self, name: str, level: int = logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._ensure_handler()
    
    def _ensure_handler(self):
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(self._create_formatter())
            self.logger.addHandler(handler)
    
    def _create_formatter(self) -> logging.Formatter:
        class ContextFormatter(logging.Formatter):
            def format(self, record):
                record.correlation_id = correlation_id.get("N/A")
                return super().format(record)
        return ContextFormatter("%(asctime)s - %(levelname)s - [%(name)s] - [%(correlation_id)s] - %(message)s")
    
    def error(self, msg: str, **kwargs):
        self.logger.error(msg, extra={"context": kwargs})

# logging/interceptors.py
class LoggingInterceptor:
    """Decoupled logging that doesn't affect business logic."""
    
    @staticmethod
    def log_exception(func):
        """Decorator for automatic exception logging."""
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except RLMError as e:
                logger = get_logger("exceptions")
                logger.error(
                    f"RLM Exception: {e.error_code}",
                    error_code=e.error_code,
                    correlation_id=e.correlation_id,
                    context=e.context
                )
                raise
        return wrapper
```

### 1.4 Circuit Breaker Component

**Purpose:** Prevent cascade failures by detecting patterns and halting operations.

**Key Responsibilities:**
- Monitor failure rates
- Maintain state (CLOSED, OPEN, HALF_OPEN)
- Enforce failure thresholds
- Manage recovery windows
- Provide fallback mechanisms

**Example Structure:**

```python
# circuitbreaker/base.py
from enum import Enum
from typing import Callable, Optional
import time

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking calls
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3

class CircuitBreaker:
    """Abstract circuit breaker interface."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._callbacks: list[Callable] = []
    
    def on_state_change(self, callback: Callable[['CircuitBreaker', CircuitState, CircuitState], None]):
        """Register state change callback."""
        self._callbacks.append(callback)
    
    def _transition(self, old_state: CircuitState, new_state: CircuitState):
        self.state = new_state
        for cb in self._callbacks:
            cb(self, old_state, new_state)
    
    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._failure_count >= self.config.failure_threshold:
            self._transition(CircuitState.CLOSED, CircuitState.OPEN)
    
    def record_success(self):
        self._failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self._transition(CircuitState.HALF_OPEN, CircuitState.CLOSED)

# circuitbreaker/implementation.py
class PyBreakerAdapter(CircuitBreaker):
    """Adapter for pybreaker library."""
    
    def __init__(self, pybreaker_instance, name: str, config: CircuitBreakerConfig):
        super().__init__(name, config)
        self._breaker = pybreaker_instance
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator-based usage."""
        def wrapper(*args, **kwargs):
            with self._breaker:
                return func(*args, **kwargs)
        return wrapper
```

### 1.5 Boundary Definition Matrix

| Aspect | Error Handling | Logging | Circuit Breaker |
|--------|---------------|---------|-----------------|
| **Trigger** | Exception raised | Any event | Threshold exceeded |
| **Action** | Propagate/catch | Emit structured data | Block/allow calls |
| **Timing** | Synchronous | Fire-and-forget | Stateful monitoring |
| **Configuration** | Error definitions | Output sinks | Thresholds/timeouts |
| **Dependencies** | None (base) | Context providers | Error handling |
| **Test Focus** | Exception paths | Log output | State transitions |

---

## 2. Data Flow for Error Propagation Through Async Systems

### 2.1 Async Error Flow Architecture

```
User Request
    |
    v
+--------------------------+
|  API Layer (FastAPI)     |
|  - Request validation     |
|  - Exception handlers     |
+--------------------------+
    |
    | (async call)
    v
+--------------------------+
|  Agent Layer              |
|  - Execution loop         |
|  - Async error recovery   |
+--------------------------+
    |
    | (await)
    v
+--------------------------+
|  Service Layer            |
|  - Business logic         |
|  - Circuit breaker guard  |
+--------------------------+
    |
    | (await)
    v
+--------------------------+
|  Infrastructure Layer     |
|  - DB operations          |
|  - LLM calls             |
|  - MCP tool execution     |
+--------------------------+
```

### 2.2 Error Propagation Patterns

#### Pattern 1: Exception Translation

```python
# layers/api/exception_handlers.py
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

async def rlm_exception_handler(request: Request, exc: RLMError):
    """Convert RLM exceptions to HTTP responses."""
    return JSONResponse(
        status_code=_map_error_to_status(exc),
        content={
            "error_code": exc.error_code,
            "message": exc.message,
            "correlation_id": exc.correlation_id,
            "recovery_hint": getattr(exc, "recovery_hint", None)
        }
    )

def _map_error_to_status(error: RLMError) -> int:
    if isinstance(error, GuardrailError):
        return 400  # Bad Request
    elif isinstance(error, CircuitOpenError):
        return 503  # Service Unavailable
    return 500
```

#### Pattern 2: Circuit Breaker Integration

```python
# layers/agent/circuit_guard.py
from functools import wraps
from circuitbreaker import circuit_breaker

class CircuitGuard:
    """Circuit breaker integration for async operations."""
    
    @staticmethod
    def protect(service_name: str, config: CircuitBreakerConfig):
        """Decorator to wrap async functions with circuit breaker."""
        def decorator(func: Callable):
            breaker = get_breaker(service_name, config)
            
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if breaker.state == CircuitState.OPEN:
                    raise CircuitOpenError(
                        f"Circuit {service_name} is open",
                        correlation_id=correlation_id.get()
                    )
                
                try:
                    result = await func(*args, **kwargs)
                    breaker.record_success()
                    return result
                except Exception as e:
                    breaker.record_failure()
                    raise
            return wrapper
        return decorator

# Usage
@CircuitGuard.protect("llm_service", LLMConfig)
async def call_llm(prompt: str) -> LLMResponse:
    """LLM call with circuit breaker protection."""
    return await llm.generate(prompt)
```

#### Pattern 3: Async Context Propagation

```python
# layers/core/context_propagation.py
import asyncio
from contextvars import ContextVar
from typing import Optional

class ErrorContext:
    """Context-aware error propagation."""
    
    correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id")
    trace_stack: ContextVar[list] = ContextVar("trace_stack", default_factory=list)
    
    @classmethod
    def start_span(cls, operation_name: str) -> str:
        """Start a new trace span."""
        span_id = generate_span_id()
        current_stack = cls.trace_stack.get()
        current_stack.append({
            "operation": operation_name,
            "span_id": span_id,
            "start_time": time.now()
        })
        return span_id
    
    @classmethod
    def capture_exception(cls, error: Exception) -> dict:
        """Capture exception with full context."""
        return {
            "error": error,
            "correlation_id": cls.correlation_id.get(),
            "trace_stack": cls.trace_stack.get(),
            "timestamp": time.now()
        }
```

### 2.3 Complete Async Flow Diagram

```
1. REQUEST START
   API Layer receives request
   -> Generates correlation_id
   -> Starts trace span
   -> Passes correlation_id via ContextVar

2. AGENT EXECUTION
   Agent.query() called with correlation_id
   -> Creates execution context
   -> Wraps operations with error handlers
   -> Logs progress

3. SERVICE CALLS
   Business logic calls services
   -> Circuit breaker checks state
   -> If OPEN: CircuitOpenError raised
   -> If CLOSED/HALF_OPEN: Proceeds
   -> Failure recorded in breaker

4. ERROR DETECTION
   Any exception caught at appropriate level
   -> RLMError hierarchy applied
   -> Context added (correlation_id, span_id)
   -> Logged with full context
   -> Transformed if crossing layer boundary

5. ERROR RECOVERY
   Recovery strategies applied:
   -> GuardrailError: User input correction
   -> CircuitOpenError: Fallback path
   -> TransientError: Retry with backoff
   -> PermanentError: Fail fast

6. RESPONSE RETURN
   Success or error returned upward
   -> Correlation ID preserved
   -> Trace span closed
   -> Final log emitted
   -> Response sent to user
```

### 2.4 Error Flow Data Structure

```python
# Error propagation envelope
class ErrorEnvelope:
    """Standard error format across all layers."""
    
    def __init__(
        self,
        error_code: str,
        message: str,
        correlation_id: str,
        layer: str,
        timestamp: datetime,
        recovery_hint: Optional[str] = None,
        trace_stack: Optional[list] = None
    ):
        self.error_code = error_code
        self.message = message
        self.correlation_id = correlation_id
        self.layer = layer
        self.timestamp = timestamp
        self.recovery_hint = recovery_hint
        self.trace_stack = trace_stack or []
        self._original_error: Optional[Exception] = None
    
    def wrap(self, original_error: Exception) -> 'ErrorEnvelope':
        """Wrap an exception in this envelope."""
        self._original_error = original_error
        return self
    
    def to_dict(self) -> dict:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "correlation_id": self.correlation_id,
            "layer": self.layer,
            "timestamp": self.timestamp.isoformat(),
            "recovery_hint": self.recovery_hint,
            "trace_stack": self.trace_stack,
            "cause": str(self._original_error) if self._original_error else None
        }
```

---

## 3. Build Order Dependencies

### 3.1 Dependency Hierarchy

```
Layer 0: Foundation (No dependencies)
├── Python Standard Library
├── Type system (typing module)
└── Basic utilities (contextvars, dataclasses)

Layer 1: Infrastructure
├── Configuration system
│   └── Settings management (.env, config files)
├── Logging framework
│   └── Logger initialization, formatters
└── Error definitions
    └── Base exception classes

Layer 2: Core Abstractions
├── Circuit breaker base
│   └── Requires: Error handling, logging
├── Context propagation
│   └── Requires: Error definitions
└── State management
    └── Requires: Configuration, context

Layer 3: Business Logic
├── Agent core
│   └── Requires: State management, circuit breaker, context
├── Service layer
│   └── Requires: Circuit breaker, error handling
└── Validation/guardrails
    └── Requires: Error definitions, logging

Layer 4: Integration
├── LLM integration
│   └── Requires: Service layer, circuit breaker
├── Graph database layer
│   └── Requires: Configuration, error handling
└── MCP integration
    └── Requires: Service layer, isolation

Layer 5: API & Presentation
├── REST API (FastAPI)
│   └── Requires: Business logic, error handlers
├── WebSocket streaming
│   └── Requires: Logging, context propagation
└── CLI interface
    └── Requires: Agent core, configuration
```

### 3.2 Implementation Order

#### Phase 1: Foundation (Must complete first)

```python
# 1.1 Configuration
# File: src/core/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings - no external dependencies."""
    database_url: str = "localhost:6379"
    llm_api_key: Optional[str] = None
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"

# 1.2 Error base classes
# File: src/exceptions/base.py
class RLMError(Exception):
    """Base error class."""
    error_code: str = "RLM_BASE"
    
    def __init__(self, message: str, **context):
        self.message = message
        self.context = context
        super().__init__(f"[{self.error_code}] {message}")

# 1.3 Logger initialization
# File: src/logging/logger.py
import logging
import sys

def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Basic logger factory."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
```

#### Phase 2: Core Abstractions (Requires Phase 1)

```python
# 2.1 Circuit breaker state machine
# File: src/circuitbreaker/base.py
from enum import Enum
from typing import Callable, Optional
from src.exceptions.base import RLMError

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitOpenError(RLMError):
    """Raised when circuit is open."""
    error_code = "CIRCUIT_OPEN"

class CircuitBreaker:
    """Circuit breaker with state machine."""
    
    def __init__(self, name: str, failure_threshold: int = 5):
        self.name = name
        self.failure_threshold = failure_threshold
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self._callbacks: list[Callable] = []
    
    def on_state_change(self, callback: Callable):
        """Register state change callback."""
        self._callbacks.append(callback)
```

#### Phase 3: Business Logic (Requires Phase 2)

```python
# 3.1 Agent core with error handling
# File: src/core/agent.py
from typing import Any, Dict, Optional
from src.exceptions.base import RLMError
from src.circuitbreaker.base import CircuitBreaker, CircuitOpenError
from src.logging.logger import get_logger

logger = get_logger("agent")

class Agent:
    """Agent with circuit breaker integration."""
    
    def __init__(self, circuit_breaker: CircuitBreaker):
        self.circuit = circuit_breaker
    
    async def query(self, prompt: str) -> Dict[str, Any]:
        """Execute query with full error handling."""
        if self.circuit.state == CircuitState.OPEN:
            raise CircuitOpenError(
                f"Circuit {self.circuit.name} is open",
                retry_after=60
            )
        
        try:
            result = await self._execute(prompt)
            self.circuit.failure_count = 0
            return result
        except Exception as e:
            self.circuit.failure_count += 1
            if self.circuit.failure_count >= self.circuit.failure_threshold:
                self.circuit.state = CircuitState.OPEN
            raise
```

#### Phase 4: Integration (Requires Phase 3)

```python
# 4.1 LLM integration with all patterns
# File: src/integrations/llm.py
from src.core.agent import Agent
from src.circuitbreaker.base import CircuitBreaker

class LLMService:
    """LLM service with circuit breaker and error handling."""
    
    def __init__(self, agent: Agent):
        self.agent = agent
        self.circuit = CircuitBreaker("llm", failure_threshold=3)
    
    async def generate(self, prompt: str) -> str:
        """Generate with full error handling."""
        try:
            return await self._call_llm(prompt)
        except RLMError:
            raise
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise LLMError(str(e))
```

### 3.3 Dependency Verification

```python
# scripts/verify_dependencies.py
"""
Verify build order dependencies are satisfied.
"""

ORDER = [
    ("Configuration", ["src/core/config.py"]),
    ("Error Base", ["src/exceptions/base.py"]),
    ("Logger", ["src/logging/logger.py"]),
    ("Circuit Breaker", ["src/circuitbreaker/base.py"]),
    ("Context", ["src/core/context.py"]),
    ("Agent Core", ["src/core/agent.py"]),
    ("LLM Integration", ["src/integrations/llm.py"]),
    ("API Layer", ["src/api/main.py"]),
]

def verify():
    """Verify all dependencies are in correct order."""
    import ast
    import os
    
    def get_imports(filepath: str) -> set:
        imports = set()
        with open(filepath) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
        return imports
    
    for layer_name, files in ORDER:
        layer_imports = set()
        for filepath in files:
            if os.path.exists(filepath):
                layer_imports.update(get_imports(filepath))
        
        print(f"{layer_name}: {layer_imports}")

if __name__ == "__main__":
    verify()
```

---

## 4. Test Infrastructure for Large Legacy Files

### 4.1 Legacy File Challenges

| Challenge | Description | Mitigation Strategy |
|-----------|-------------|---------------------|
| **Size** | Files > 2000 lines | Split into test modules |
| **Coupling** | High interdependencies | Mock external dependencies |
| **Side Effects** | Global state | Reset state between tests |
| **Old Patterns** | No type hints | Use runtime verification |
| **Coverage Gaps** | Untested code | Incremental coverage |

### 4.2 Test File Organization

```
tests/
├── conftest.py                    # Global fixtures
├── legacy/
│   ├── test_legacy_agent.py      # Agent tests
│   │   ├── TestAgentCore
│   │   ├── TestAgentExecution
│   │   └── TestAgentRecovery
│   ├── test_legacy_dream.py      # Dreamer tests
│   │   ├── TestDreamAnalysis
│   │   └── TestDreamConsolidation
│   └── test_legacy_sheaf.py     # Sheaf tests
│       ├── TestSheafTopology
│       └── TestSheafConsistency
├── integration/
│   ├── test_agent_dream.py       # Cross-component
│   ├── test_agent_sheaf.py
│   └── test_full_pipeline.py
├── mocking/
│   ├── mocks.py                  # Common mocks
│   └── fixtures.py               # Test data
└── utils/
    ├── coverage_report.py        # Coverage analysis
    └── test_runner.py            # Custom test runner
```

### 4.3 Mocking Strategy for Legacy Files

#### Step 1: Identify Dependencies

```python
# scripts/analyze_dependencies.py
"""
Analyze legacy file dependencies for mocking.
"""

import ast
from pathlib import Path

def analyze_file(filepath: str) -> dict:
    """Analyze a file's dependencies."""
    with open(filepath) as f:
        tree = ast.parse(f.read())
    
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    
    # Categorize by type
    external = [i for i in imports if not i.startswith('.')]
    internal = [i for i in imports if i.startswith('.')]
    
    return {
        "file": filepath,
        "external_deps": external,
        "internal_deps": internal,
        "line_count": len(tree.body)
    }

# Run analysis
for filepath in Path("src/").rglob("*.py"):
    result = analyze_file(str(filepath))
    if result["line_count"] > 1000:
        print(f"LEGACY FILE: {filepath}")
        print(f"  External deps: {len(result['external_deps'])}")
        print(f"  Internal deps: {len(result['internal_deps'])}")
```

#### Step 2: Create Mock Library

```python
# tests/mocking/mocks.py
"""
Mock library for legacy file testing.
"""

import sys
from unittest.mock import MagicMock, AsyncMock, patch
from types import ModuleType
from typing import Any

class MockRegistry:
    """Registry for mocked modules."""
    
    def __init__(self):
        self.mocks: dict[str, ModuleType] = {}
    
    def mock_external(self, module_name: str) -> ModuleType:
        """Mock an external dependency."""
        mock = ModuleType(f"mocked_{module_name}")
        sys.modules[module_name] = mock
        self.mocks[module_name] = mock
        return mock
    
    def mock_database(self) -> ModuleType:
        """Mock database module."""
        mock = self.mock_external("falkordb")
        mock.GraphClient = MagicMock()
        return mock
    
    def mock_llm_service(self) -> ModuleType:
        """Mock LLM service."""
        mock = self.mock_external("graph_rlm.backend.src.core.llm")
        mock.llm = MagicMock()
        mock.llm.generate = AsyncMock(return_value={"response": "test"})
        return mock
    
    def mock_sheaf(self) -> ModuleType:
        """Mock sheaf monitor."""
        mock = self.mock_external("graph_rlm.backend.src.core.sheaf")
        mock.sheaf = MagicMock()
        mock.sheaf.diagnose_trace = MagicMock(return_value={"status": "HEALTHY"})
        return mock
    
    def reset(self):
        """Reset all mocks."""
        for module_name, mock in self.mocks.items():
            if module_name in sys.modules:
                del sys.modules[module_name]
        self.mocks.clear()

mock_registry = MockRegistry()

def setup_legacy_mocks():
    """Setup all mocks for legacy file testing."""
    mock_registry.mock_database()
    mock_registry.mock_llm_service()
    mock_registry.mock_sheaf()
```

#### Step 3: Test Class Structure for Legacy Files

```python
# tests/legacy/test_legacy_agent.py
"""
Test suite for legacy agent.py file.
Tests organized by functionality, not file structure.
"""

import sys
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from contextlib import asynccontextmanager

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup mocks BEFORE importing legacy modules
from tests.mocking.mocks import setup_legacy_mocks, mock_registry
setup_legacy_mocks()

# Now import legacy modules
from graph_rlm.backend.src.core.agent import Agent
from graph_rlm.backend.src.core.state import ExecutionState


class TestLegacyAgentCore(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for Agent core functionality.
    
    Tests cover:
    - TC1: Agent initialization
    - TC2: Query execution
    - TC3: Error handling
    - TC4: Recovery patterns
    """
    
    async def asyncSetUp(self):
        """Setup before each test."""
        # Reset mocks
        mock_registry.reset()
        setup_legacy_mocks()
        
        # Create fresh instance
        self.agent = Agent()
        self.agent.session_cache = {}
        
        # Setup test fixtures
        self.test_session_id = "test_session_001"
    
    async def asyncTearDown(self):
        """Cleanup after each test."""
        pass
    
    # ====== TC1: Initialization Tests ======
    
    async def test_agent_initialization(self):
        """TC1-01: Agent initializes with all components."""
        agent = Agent()
        
        self.assertIsNotNone(agent.db)
        self.assertIsNotNone(agent.llm)
        self.assertIsNotNone(agent.runtime)
        self.assertIsNotNone(agent.navigator)
    
    async def test_agent_with_custom_config(self):
        """TC1-02: Agent accepts custom configuration."""
        with patch("graph_rlm.backend.src.core.agent.get_config") as mock_config:
            mock_config.return_value = MagicMock()
            agent = Agent()
            self.assertIsNotNone(agent.config)
    
    # ====== TC2: Query Execution Tests ======
    
    async def test_query_execution_happy_path(self):
        """TC2-01: Query executes successfully."""
        # Setup mocks
        self.agent.llm.generate = AsyncMock(return_value="response")
        self.agent.db.query = MagicMock(return_value={"thoughts": []})
        
        # Execute
        result = await self.agent.query("test prompt")
        
        # Verify
        self.assertEqual(result["status"], "success")
        self.agent.llm.generate.assert_called_once()
    
    async def test_query_with_context(self):
        """TC2-02: Query loads context from scratchpad."""
        self.agent.scratchpad_builder = MagicMock()
        self.agent.scratchpad_builder.build = AsyncMock(return_value={"context": "data"})
        
        result = await self.agent.query("test", context={"load_scratchpad": True})
        
        self.assertIn("context", result)
    
    # ====== TC3: Error Handling Tests ======
    
    async def test_error_handling_invalid_input(self):
        """TC3-01: Agent handles invalid input gracefully."""
        with self.assertRaises(ValueError):
            await self.agent.query("")
    
    async def test_error_handling_db_failure(self):
        """TC3-02: Agent handles database failure."""
        self.agent.db.query = MagicMock(side_effect=ConnectionError("DB down"))
        
        result = await self.agent.query("test")
        
        self.assertEqual(result["status"], "error")
        self.assertIn("connection", result["message"].lower())
    
    async def test_error_recovery_after_db_failure(self):
        """TC3-03: Agent recovers after DB failure."""
        call_count = [0]
        
        def flaky_query(*args):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("DB down")
            return {"thoughts": []}
        
        self.agent.db.query = MagicMock(side_effect=flaky_query)
        
        result = await self.agent.query("test")
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(call_count[0], 2)
    
    # ====== TC4: Circuit Breaker Integration Tests ======
    
    async def test_circuit_breaker_open(self):
        """TC4-01: Agent respects circuit breaker state."""
        with patch("graph_rlm.backend.src.core.agent.CircuitBreaker") as mock_cb:
            mock_breaker = MagicMock()
            mock_breaker.state = "OPEN"
            mock_cb.return_value = mock_breaker
            
            result = await self.agent.query("test")
            
            self.assertEqual(result["status"], "circuit_open")
    
    # ====== TC5: Session Management Tests ======
    
    async def test_session_isolation(self):
        """TC5-01: Sessions are properly isolated."""
        session1_cache = {"key": "value1"}
        session2_cache = {"key": "value2"}
        
        self.agent.session_cache["session1"] = session1_cache
        self.agent.session_cache["session2"] = session2_cache
        
        self.assertNotEqual(
            self.agent.session_cache["session1"],
            self.agent.session_cache["session2"]
        )
```

### 4.4 Legacy File Test Utilities

```python
# tests/utils/legacy_test_helpers.py
"""
Utilities for testing legacy Python files.
"""

import sys
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock

class LegacyFileAnalyzer:
    """Analyze legacy files for test opportunities."""
    
    @staticmethod
    def get_functions(filepath: str) -> List[dict]:
        """Extract all functions from a file."""
        import ast
        
        with open(filepath) as f:
            tree = ast.parse(f.read())
        
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    "name": node.name,
                    "lineno": node.lineno,
                    "args": [arg.arg for arg in node.args.args],
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "docstring": ast.get_docstring(node)
                })
        
        return functions
    
    @staticmethod
    def get_classes(filepath: str) -> List[dict]:
        """Extract all classes from a file."""
        import ast
        
        with open(filepath) as f:
            tree = ast.parse(f.read())
        
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append({
                    "name": node.name,
                    "lineno": node.lineno,
                    "methods": [
                        method.name for method in node.body
                        if isinstance(method, ast.FunctionDef)
                    ],
                    "docstring": ast.get_docstring(node)
                })
        
        return classes
    
    @staticmethod
    def generate_test_template(filepath: str) -> str:
        """Generate test template for a legacy file."""
        classes = LegacyFileAnalyzer.get_classes(filepath)
        functions = LegacyFileAnalyzer.get_functions(filepath)
        
        template = f'''"""
Auto-generated tests for {filepath}
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class TestGenerated(unittest.TestCase):
    """Generated tests for legacy file."""
    
    def setUp(self):
        """Setup test fixtures."""
        pass
    
    # Test classes for each class in the file
'''
        
        for cls in classes:
            template += f'''
    # ====== {cls["name"]} Tests ======
'''
            for method in cls["methods"]:
                template += f'''
    def test_{cls["name"]}_{method}(self):
        """Test {cls["name"]}.{method}"""
        self.fail("Test not implemented")
'''
        
        return template


def run_legacy_coverage_analysis(filepath: str) -> Dict[str, Any]:
    """Analyze coverage for a legacy file."""
    analyzer = LegacyFileAnalyzer
    functions = analyzer.get_functions(filepath)
    classes = analyzer.get_classes(filepath)
    
    return {
        "filepath": filepath,
        "total_classes": len(classes),
        "total_functions": len(functions),
        "classes": classes,
        "functions": functions,
        "undocumented": [
            f for f in functions if not f["docstring"]
        ]
    }
```

### 4.5 Incremental Coverage Strategy

```python
# scripts/incremental_coverage.py
"""
Strategy for incrementally increasing test coverage of legacy files.
"""

COVERAGE_STRATEGY = {
    "phase_1": {
        "target": "src/core/config.py",
        "goal": "100%",
        "priority": "HIGHEST",
        "tests_needed": ["test_config_loading", "test_env_vars"]
    },
    "phase_2": {
        "target": "src/core/logger.py",
        "goal": "100%",
        "priority": "HIGHEST",
        "tests_needed": ["test_logger_initialization", "test_log_levels"]
    },
    "phase_3": {
        "target": "src/core/agent.py",
        "goal": "80%",
        "priority": "HIGH",
        "tests_needed": ["test_query_execution", "test_error_handling", "test_session_mgmt"]
    },
    "phase_4": {
        "target": "src/core/dream.py",
        "goal": "70%",
        "priority": "MEDIUM",
        "tests_needed": ["test_dream_cycle", "test_holonomy_analysis"]
    },
    "phase_5": {
        "target": "src/core/sheaf.py",
        "goal": "60%",
        "priority": "MEDIUM",
        "tests_needed": ["test_sheaf_consistency", "test_topology_checks"]
    }
}

def generate_phase_report():
    """Generate coverage report by phase."""
    import subprocess
    
    result = subprocess.run(
        ["pytest", "--cov=graph_rlm", "--cov-report=json"],
        capture_output=True,
        text=True
    )
    
    # Parse and group by phase
    pass

# Run coverage analysis
if __name__ == "__main__":
    run_legacy_coverage_analysis("graph_rlm/backend/src/core/agent.py")
```

---

## 5. Recommended Architecture Summary

### 5.1 Component Responsibilities

| Component | Responsibility | Dependencies | Output |
|-----------|---------------|--------------|--------|
| **Error Handler** | Define, raise, catch exceptions | None (foundation) | Exception objects |
| **Logger** | Capture and route diagnostic data | Error definitions | Log records |
| **Circuit Breaker** | Monitor failures, block requests | Error handling, logging | State changes, metrics |
| **Test Infrastructure** | Verify behavior, mock dependencies | All components | Test results, coverage |

### 5.2 Data Flow Summary

```
Request
  -> API Layer (exception handlers configured)
  -> Agent Layer (context propagation started)
  -> Service Layer (circuit breaker monitored)
  -> Infrastructure Layer (errors caught and logged)
  -> Response (correlation preserved)
```

### 5.3 Build Order Summary

1. **Phase 1**: Configuration, Error Base, Logger
2. **Phase 2**: Circuit Breaker, Context Propagation
3. **Phase 3**: Agent Core, Service Layer
4. **Phase 4**: Integration Layer (LLM, DB, MCP)
5. **Phase 5**: API and Presentation

### 5.4 Test Strategy Summary

| Legacy File Size | Test Approach | Mocking Level |
|-----------------|---------------|---------------|
| < 500 lines | Single test file | Minimal |
| 500-1000 lines | Multiple test classes | Moderate |
| 1000-2000 lines | Multiple test files | Extensive |
| > 2000 lines | Split legacy + new tests | Full mock library |

---

## 6. Implementation Checklist

- [ ] Create error hierarchy (`RLMError` base + specific subclasses)
- [ ] Implement structured logger with context support
- [ ] Build circuit breaker with state machine
- [ ] Configure exception handlers for each API layer
- [ ] Add correlation ID propagation throughout async flow
- [ ] Create mock library for legacy file dependencies
- [ ] Write test classes organized by functionality
- [ ] Implement incremental coverage strategy
- [ ] Verify build order dependencies are satisfied
- [ ] Add circuit breaker callbacks for logging/monitoring

---

*Document Version: 1.0*
*Last Updated: 2026-02-12*
*Status: Research Complete*
