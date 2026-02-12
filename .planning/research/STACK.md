# Python Exception Handling, Testing, and Resilience Stack

## Overview

This document outlines recommended libraries and patterns for implementing robust exception handling, comprehensive test infrastructure, and circuit breaker patterns in the graph-rlm project. All recommendations are based on Python 3.13+ compatibility and 2025 best practices.

---

## 1. Custom Exception Hierarchies with Error Codes

### Recommended Library: Pydantic with Custom Exception Framework

**Why Pydantic?** The project already depends on Pydantic (via `pydantic-settings>=2.12.0`). Pydantic's `ValidationError` provides an excellent pattern for structured error handling. For custom exception hierarchies, build a native Python solution leveraging Python 3.13's improved exception group syntax.

### Exception Hierarchy Pattern

```python
# src/core/exceptions/base.py
from enum import Enum
from typing import Optional
import traceback
import logging

class ErrorCode(Enum):
    """Centralized error codes for the application."""
    # Core errors (1xxx)
    CORE_UNKNOWN = "CORE_1000"
    CORE_INIT_FAILED = "CORE_1001"
    CORE_STATE_INVALID = "CORE_1002"
    
    # Graph errors (2xxx)
    GRAPH_NODE_NOT_FOUND = "GRAPH_2000"
    GRAPH_EDGE_CREATION_FAILED = "GRAPH_2001"
    GRAPH_QUERY_TIMEOUT = "GRAPH_2002"
    
    # Skill execution errors (3xxx)
    SKILL_NOT_FOUND = "SKILL_3000"
    SKILL_EXECUTION_FAILED = "SKILL_3001"
    SKILL_TIMEOUT = "SKILL_3002"
    SKILL_IMPORT_ERROR = "SKILL_3003"
    
    # External service errors (4xxx)
    EXTERNAL_SERVICE_UNAVAILABLE = "EXTERNAL_4000"
    EXTERNAL_RATE_LIMITED = "EXTERNAL_4001"
    EXTERNAL_TIMEOUT = "EXTERNAL_4002"

class BaseGraphRLMError(Exception):
    """Base exception for all graph-rlm errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        context: Optional[dict] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.cause = cause
        self.timestamp = __import__('datetime').datetime.utcnow()
    
    def to_dict(self) -> dict:
        """Serialize exception for logging/API responses."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code.value,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "caused_by": self.cause.__class__.__name__ if self.cause else None
        }
    
    def log(self, logger: Optional[logging.Logger] = None, level: int = logging.ERROR):
        """Log exception with full context."""
        logger = logger or logging.getLogger(__name__)
        logger.log(
            level,
            f"[{self.error_code.value}] {self.message}",
            extra={"error_data": self.to_dict()}
        )

class CoreError(BaseGraphRLMError):
    """Base class for core system errors."""
    pass

class GraphError(BaseGraphRLMError):
    """Base class for graph-related errors."""
    pass

class SkillExecutionError(BaseGraphRLMError):
    """Base class for skill execution errors."""
    pass

class ExternalServiceError(BaseGraphRLMError):
    """Base class for external service errors."""
    pass
```

### Rationale

1. **Enum-based error codes** provide discoverability and prevent string typos
2. **Hierarchical structure** allows catching at different levels (`except GraphError` catches all graph-related issues)
3. **Context dictionary** enables passing additional debugging information
4. **Cause chaining** preserves the original exception traceback
5. **Serialization support** for logging and API error responses

---

## 2. Pytest Fixtures and Mocking for Async/Await Code

### Recommended Stack

| Library | Version | Purpose |
|---------|---------|---------|
| pytest | >=9.0.2 | Core testing framework (already in project) |
| pytest-asyncio | >=1.3.0 | Async test support (already in project dev dependencies) |
| pytest-cov | >=6.0.0 | Coverage reporting |
| pytest-mock | >=3.14.0 | Mocking utilities |
| respx | >=0.21.0 | HTTP request mocking for async |
| httpx | >=0.28.0 | Async HTTP client for testing |

### Pytest Configuration

```python
# pytest.ini or pyproject.toml addition
[tool.pytest.ini_options]
asyncio_mode = "auto"  # pytest-asyncio 1.0+ feature
asyncio_default_fixture_scope = "function"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--tb=short",
    "--asyncio-mode=auto"
]
```

### Async Fixture Pattern

```python
# tests/conftest.py
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from typing import AsyncGenerator

@pytest_asyncio.fixture
async def mock_graph_db():
    """Mock graph database for isolation testing."""
    mock = MagicMock()
    mock.query = AsyncMock(return_value={"nodes": [], "edges": []})
    mock.create_node = AsyncMock(return_value={"id": "test_node"})
    mock.create_edge = AsyncMock(return_value={"id": "test_edge"})
    yield mock
    # Cleanup handled by async generator

@pytest_asyncio.fixture
async def skill_context(mock_graph_db) -> AsyncGenerator[dict, None]:
    """Shared skill execution context with mocked dependencies."""
    yield {
        "graph": mock_graph_db,
        "config": {"timeout": 30, "retries": 3},
        "namespace": {}
    }

@pytest_asyncio.fixture
def sample_skill_code():
    """Sample skill code for testing."""
    return '''
async def execute(input_data):
    return {"status": "success", "result": input_data}
'''
```

### Mocking Patterns for Async Code

```python
# tests/mocks/async_mocks.py
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

class AsyncContextManagerMock:
    """Mock for async context managers."""
    
    def __init__(self, return_value=None, side_effect=None):
        self.return_value = return_value
        self.side_effect = side_effect
    
    async def __aenter__(self):
        if self.side_effect:
            raise self.side_effect
        return self.return_value
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

def create_async_mock(*args, **kwargs):
    """Factory for creating async mocks with consistent behavior."""
    mock = AsyncMock(*args, **kwargs)
    mock.side_effect = None  # Default no exception
    return mock

# tests/skills/test_skill_execution.py
import pytest
from unittest.mock import patch, AsyncMock

class TestSkillExecution:
    """Test suite for skill execution with async/await."""
    
    @pytest.mark.asyncio
    async def test_skill_with_mocked_external_call(self, skill_context):
        """Test skill that makes external HTTP calls."""
        with patch('skills.external.http_client') as mock_client:
            mock_client.get = AsyncMock(return_value={"data": "test"})
            
            from skills.example_skill import execute
            result = await execute(skill_context)
            
            assert result["status"] == "success"
            mock_client.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_skill_handles_timeout(self, skill_context):
        """Test skill timeout handling."""
        with pytest.raises(SkillExecutionError) as exc_info:
            from skills.slow_skill import execute
            await execute(skill_context, timeout=0.001)
        
        assert exc_info.value.error_code == ErrorCode.SKILL_TIMEOUT
    
    @pytest.mark.asyncio
    async def test_skill_with_retry_logic(self, skill_context):
        """Test skill retry mechanism using respx."""
        import respx
        from httpx import AsyncClient
        
        @respx.mock
        async def test_with_mocked_retries():
            # Mock endpoint that fails twice then succeeds
            call_count = 0
            
            async def handler(request):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    return httpx.Response(503)
                return httpx.Response(200, json={"success": True})
            
            respx.get("http://test/api").mock(side_effect=handler)
            
            client = AsyncClient()
            # Test retry logic
            await retry_operation(client.get("http://test/api"))
            assert call_count == 3
```

### Rationale

1. **`asyncio_mode = "auto"`** - Eliminates repetitive `@pytest.mark.asyncio` decorators
2. **`pytest-asyncio`** - Properly handles async fixtures and test cleanup
3. **`pytest-mock`** - Provides consistent `.mock` fixture across all tests
4. **`respx`** - Best-in-class HTTP mocking for both sync and async, works with `httpx`
5. **Custom `AsyncContextManagerMock`** - Handles async context managers which are tricky to mock

---

## 3. Circuit Breaker Pattern Implementations

### Recommended Library: PyCircuitBreaker or Custom Implementation

**Status**: No widely-adopted async-aware circuit breaker library exists that meets all requirements. **Recommendation: Implement a custom async-aware circuit breaker** leveraging Python 3.13's `asyncio` improvements.

### Custom Circuit Breaker Implementation

```python
# src/core/resilience/circuit_breaker.py
from enum import Enum
from typing import Callable, Any, Optional
from dataclasses import dataclass, field
import asyncio
import time
from functools import wraps

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject all calls
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 2         # Successes in half-open to close
    timeout_seconds: float = 60.0       # Time in open state before half-open
    expected_exception: type = Exception
    monitoring_window: int = 30         # Seconds to track failures
    
    # Callbacks
    on_state_change: Optional[Callable] = None
    on_failure: Optional[Callable] = None
    on_success: Optional[Callable] = None

@dataclass
class CircuitStats:
    """Statistics for circuit breaker."""
    failures: int = 0
    successes: int = 0
    last_failure_time: Optional[float] = None
    total_calls: int = 0
    rejected_calls: int = 0
    
    def record_failure(self):
        self.failures += 1
        self.total_calls += 1
        self.last_failure_time = time.time()
    
    def record_success(self):
        self.successes += 1
        self.total_calls += 1

class CircuitBreaker:
    """
    Async-aware circuit breaker implementation.
    
    Usage:
        @circuit_breaker
        async def fragile_service_call():
            return await external_api.request()
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self._state_lock = asyncio.Lock()
        self._half_open_successes = 0
        self._opened_at: Optional[float] = None
    
    def _notify_state_change(self, old_state: CircuitState, new_state: CircuitState):
        if self.config.on_state_change:
            self.config.on_state_change(
                self.name, old_state, new_state, self.stats
            )
    
    def _notify_failure(self, error: Exception):
        if self.config.on_failure:
            self.config.on_failure(self.name, error)
    
    def _notify_success(self):
        if self.config.on_success:
            self.config.on_success(self.name)
    
    async def _attempt_open(self):
        """Transition to OPEN state."""
        async with self._state_lock:
            if self.state == CircuitState.CLOSED:
                old_state = self.state
                self.state = CircuitState.OPEN
                self._opened_at = time.time()
                self._notify_state_change(old_state, CircuitState.OPEN)
    
    async def def _attempt_close(self):
        """Transition to CLOSED state."""
        async with self._state_lock:
            if self.state == CircuitState.HALF_OPEN:
                old_state = self.state
                self.state = CircuitState.CLOSED
                self.stats = CircuitStats()  # Reset stats
                self._half_open_successes = 0
                self._notify_state_change(old_state, CircuitState.CLOSED)
    
    async def _check_timeout(self) -> bool:
        """Check if timeout expired, transition to half-open."""
        if self.state == CircuitState.OPEN:
            if self._opened_at and (time.time() - self._opened_at) >= self.config.timeout_seconds:
                async with self._state_lock:
                    if self.state == CircuitState.OPEN:
                        self.state = CircuitState.HALF_OPEN
                        self._notify_state_change(CircuitState.OPEN, CircuitState.HALF_OPEN)
                        return True
        return False
    
    async def __call__(self, func: Callable) -> Callable:
        """Decorator usage of circuit breaker."""
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            async with self._state_lock:
                await self._check_timeout()
                
                if self.state == CircuitState.OPEN:
                    self.stats.rejected_calls += 1
                    raise CircuitOpenError(
                        f"Circuit {self.name} is OPEN. Service unavailable.",
                        retry_after=self.config.timeout_seconds
                    )
            
            try:
                result = await func(*args, **kwargs)
                self.stats.record_success()
                self._notify_success()
                
                if self.state == CircuitState.HALF_OPEN:
                    self._half_open_successes += 1
                    if self._half_open_successes >= self.config.success_threshold:
                        await self._attempt_close()
                
                return result
                
            except self.config.expected_exception as e:
                self.stats.record_failure()
                self._notify_failure(e)
                
                if self.state == CircuitState.CLOSED:
                    if self.stats.failures >= self.config.failure_threshold:
                        await self._attempt_open()
                
                elif self.state == CircuitState.HALF_OPEN:
                    # Any failure in half-open goes back to open
                    await self._attempt_open()
                
                raise
        
        return wrapper

class SyncCircuitBreaker:
    """Synchronous circuit breaker for non-async operations."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self._state_lock = asyncio.Lock() if False else None  # For consistency
        self._opened_at: Optional[float] = None
        import threading
        self._lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with self._lock:
                if self.state == CircuitState.OPEN:
                    if self._opened_at and (time.time() - self._opened_at) >= self.config.timeout_seconds:
                        self.state = CircuitState.HALF_OPEN
                    else:
                        self.stats.rejected_calls += 1
                        raise CircuitOpenError(
                            f"Circuit {self.name} is OPEN",
                            retry_after=self.config.timeout_seconds
                        )
            
            try:
                result = func(*args, **kwargs)
                self.stats.record_success()
                
                with self._lock:
                    if self.state == CircuitState.HALF_OPEN:
                        self.state = CircuitState.CLOSED
                        self.stats = CircuitStats()
                
                return result
                
            except self.config.expected_exception as e:
                self.stats.record_failure()
                with self._lock:
                    if self.state == CircuitState.CLOSED:
                        if self.stats.failures >= self.config.failure_threshold:
                            self.state = CircuitState.OPEN
                            self._opened_at = time.time()
                raise
        
        return wrapper

class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    
    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after

# Usage examples
circuit_config = CircuitBreakerConfig(
    failure_threshold=3,
    timeout_seconds=30,
    expected_exception=(ConnectionError, TimeoutError),
    on_state_change=lambda name, old, new, stats: print(f"Circuit {name}: {old} -> {new}")
)

@CircuitBreaker("external-api", circuit_config)
async def call_external_service(data: dict) -> dict:
    """External API call with circuit breaker protection."""
    async with httpx.AsyncClient() as client:
        response = await client.post("https://api.example.com/data", json=data)
        response.raise_for_status()
        return response.json()
```

### Testing Circuit Breakers

```python
# tests/resilience/test_circuit_breaker.py
import pytest
from unittest.mock import AsyncMock, patch
import time

class TestCircuitBreaker:
    """Comprehensive circuit breaker tests."""
    
    @pytest.fixture
    def failing_function(self):
        """Function that fails a specified number of times."""
        call_count = 0
        
        async def func():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise ConnectionError("Service unavailable")
            return "success"
        
        return func, lambda: setattr(import_module(__name__), 'call_count', 0)
    
    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self, failing_function):
        """Circuit should open after configured failure threshold."""
        from core.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
        
        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=1,
            expected_exception=ConnectionError
        )
        
        breaker = CircuitBreaker("test", config)
        wrapped_func = breaker(failing_function[0])
        
        # First 3 calls should fail
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await wrapped_func()
        
        assert breaker.state == CircuitState.OPEN
        
        # Next call should be rejected
        with pytest.raises(CircuitOpenError):
            await wrapped_func()
    
    @pytest.mark.asyncio
    async def test_circuit_recovers_after_timeout(self, failing_function):
        """Circuit should transition to half-open after timeout."""
        from core.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
        
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=0.1,
            expected_exception=ConnectionError
        )
        
        breaker = CircuitBreaker("test", config)
        wrapped_func = breaker(failing_function[0])
        
        # Trigger open
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await wrapped_func()
        
        assert breaker.state == CircuitState.OPEN
        
        # Wait for timeout
        await asyncio.sleep(0.15)
        
        # Next call should try (half-open)
        with pytest.raises(ConnectionError):
            await wrapped_func()
```

### Rationale

1. **Custom implementation** - No async-aware library matches Python 3.13's asyncio capabilities
2. **State machine** - Clear CLOSED -> OPEN -> HALF_OPEN transitions
3. **Statistics tracking** - Enables monitoring and alerting
4. **Callbacks** - Integration points for logging and alerting
5. **Both sync and async** - Support for different operation types

---

## 4. Centralized Error Logging Patterns

### Recommended Stack

| Library | Version | Purpose |
|---------|---------|---------|
| structlog | >=24.2.0 | Structured logging with JSON output |
| loguru | >=0.7.2 | Enhanced logging with better defaults |
| python-json-logger | >=2.0.7 | JSON formatting for log aggregation |

**Recommendation**: Use **structlog** for structured logging with JSON output.

### Structured Logging Setup

```python
# src/core/logging/config.py
import structlog
import logging
import sys

def setup_logging(
    level: str = "INFO",
    json_output: bool = True,
    service_name: str = "graph-rlm"
):
    """Initialize structured logging for the application."""
    
    # Configure root logger
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )
    
    # Processor chain for structlog
    processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

# Create module-level logger
log = structlog.get_logger()

# Example usage with error tracking
import traceback

async def safe_execute_with_logging(
    operation: str,
    func,
    *args,
    error_code: Optional[ErrorCode] = None,
    **kwargs
):
    """Execute function with comprehensive error logging."""
    
    log.info(
        "operation_start",
        operation=operation,
        args_summary=str(args)[:100]  # Truncate for privacy
    )
    
    try:
        result = await func(*args, **kwargs)
        log.info(
            "operation_success",
            operation=operation,
            duration_ms=0  # Calculate actual duration
        )
        return result
        
    except BaseGraphRLMError as e:
        log.error(
            "operation_failure",
            operation=operation,
            error_type=e.__class__.__name__,
            error_code=e.error_code.value,
            error_message=str(e),
            context=e.context,
            error_data=e.to_dict()
        )
        raise
        
    except Exception as e:
        log.error(
            "unexpected_error",
            operation=operation,
            error_type=type(e).__name__,
            error_message=str(e),
            traceback=traceback.format_exc()
        )
        raise BaseGraphRLMError(
            message=f"Unexpected error in {operation}: {str(e)}",
            error_code=ErrorCode.CORE_UNKNOWN,
            cause=e
        )
```

### Exception Logging Integration

```python
# src/core/logging/exception_handler.py
from functools import wraps
import asyncio
from typing import Callable
import structlog

log = structlog.get_logger()

def log_exceptions(
    operation_name: str,
    reraise_as: type = None,
    extra_fields: dict = None
):
    """Decorator for comprehensive exception logging."""
    
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                log.error(
                    f"{operation_name}_failed",
                    operation=operation_name,
                    exception_type=type(e).__name__,
                    exception_message=str(e),
                    **(extra_fields or {})
                )
                if reraise_as:
                    raise reraise_as(str(e)) from e
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log.error(
                    f"{operation_name}_failed",
                    operation=operation_name,
                    exception_type=type(e).__name__,
                    exception_message=str(e),
                    **(extra_fields or {})
                )
                if reraise_as:
                    raise reraise_as(str(e)) from e
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

# Usage
@log_exceptions("skill_execution", reraise_as=SkillExecutionError)
async def execute_skill(skill_name: str, input_data: dict):
    """Execute skill with automatic error logging."""
    # ... skill execution logic
    pass
```

### Centralized Error Handler

```python
# src/core/logging/centralized_handler.py
import asyncio
import logging
from typing import Optional
from dataclasses import dataclass
from enum import Enum

class ErrorSeverity(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class ErrorReport:
    """Structured error report for centralized handling."""
    timestamp: str
    error_code: str
    severity: ErrorSeverity
    component: str
    message: str
    context: dict
    traceback: Optional[str]
    incident_id: str

class CentralizedErrorHandler:
    """
    Centralized handler for error reporting, aggregation, and response.
    """
    
    def __init__(self, service_name: str = "graph-rlm"):
        self.service_name = service_name
        self._error_counts: dict[str, int] = {}
        self._incident_counter = 0
    
    def handle(
        self,
        error: BaseGraphRLMError,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[dict] = None
    ) -> ErrorReport:
        """Handle an exception and produce a structured report."""
        
        self._incident_counter += 1
        incident_id = f"{self.service_name}-{self._incident_counter:06d}"
        
        # Track error frequency
        error_key = error.error_code.value
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
        
        report = ErrorReport(
            timestamp=__import__('datetime').datetime.utcnow().isoformat(),
            error_code=error.error_code.value,
            severity=severity,
            component=self._extract_component(error.error_code.value),
            message=error.message,
            context={**(error.context or {}), **(context or {})},
            traceback=__import__('traceback').format_exc(),
            incident_id=incident_id
        )
        
        # Log the error
        self._log_report(report)
        
        # Trigger alerts for critical errors
        if severity in (ErrorSeverity.CRITICAL, ErrorSeverity.ERROR):
            self._check_alerting_conditions(error_key, report)
        
        return report
    
    def _extract_component(self, error_code: str) -> str:
        """Extract component name from error code prefix."""
        return error_code.split('_')[0].lower()
    
    def _log_report(self, report: ErrorReport):
        """Log the error report."""
        log = structlog.get_logger()
        log_method = getattr(log, report.severity.value.lower())
        log_method(
            "error_report",
            incident_id=report.incident_id,
            error_code=report.error_code,
            component=report.component,
            message=report.message,
            context=report.context
        )
    
    def _check_alerting_conditions(self, error_key: str, report: ErrorReport):
        """Check if error rate triggers alerting."""
        # Simple threshold: more than 5 of same error in 1 minute
        count = self._error_counts.get(error_key, 0)
        if count >= 5:
            log = structlog.get_logger()
            log.warning(
                "error_rate_threshold_exceeded",
                error_code=error_key,
                count=count,
                incident_id=report.incident_id
            )

# Global handler instance
error_handler = CentralizedErrorHandler()
```

### Rationale

1. **structlog** - Best-in-class structured logging with excellent JSON support
2. **Consistent format** - All logs are JSON-structured for easy parsing
3. **Error codes** - Integrated with exception hierarchy for traceable errors
4. **Severity levels** - Enables proper alert routing
5. **Incident tracking** - Unique IDs for error correlation across services

---

## Summary of Recommendations

| Category | Library/Pattern | Version | Priority |
|----------|----------------|---------|----------|
| Exception Hierarchy | Custom Pydantic-based | N/A | Required |
| Testing Framework | pytest + pytest-asyncio | 9.0+/1.3+ | Existing |
| Async Mocking | pytest-mock + respx | 3.14+/0.21+ | Required |
| HTTP Testing | httpx | 0.28+ | Recommended |
| Circuit Breaker | Custom async-aware | N/A | Required |
| Structured Logging | structlog | 24.2+ | Required |
| JSON Logging | python-json-logger | 2.0.7+ | Optional |

### Migration Steps

1. **Create exception hierarchy** in `src/core/exceptions/`
2. **Configure structlog** in `src/core/logging/`
3. **Implement circuit breaker** in `src/core/resilience/`
4. **Update pytest configuration** with asyncio auto mode
5. **Add centralized error handler** for production monitoring
6. **Write integration tests** for all new patterns
