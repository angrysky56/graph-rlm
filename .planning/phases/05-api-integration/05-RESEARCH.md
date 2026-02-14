# Phase 5: API and Integration - Research

**Researched:** 2026-02-12
**Domain:** FastAPI exception handling, HTTP status codes, pytest-cov, MCP integration
**Confidence:** HIGH

## Summary

Phase 5 completes the Graph-RLM engineering health initiative by implementing FastAPI exception handlers for the exception hierarchy, validating MCP server circuit breaker protection, running comprehensive coverage reports, and confirming the full test suite passes. The phase builds on all prior phases (exception hierarchy, circuit breakers, test infrastructure, business logic integration).

Key findings: FastAPI's exception handling requires App.add_exception_handler() for custom exception classes. HTTP status codes map to exception hierarchy (4xx for ValidationError, 5xx for others). pytest-cov can generate incremental reports via --cov. MCP server validation requires testing circuit open scenarios.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| FastAPI | >=0.128.0 | Web framework | Already in dependencies |
| pytest-cov | >=6.0.0 | Coverage reporting | Already in Phase 3 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| httpx | >=0.27.0 | HTTP client for API testing | Test client creation |
| requests | any | HTTP library | Legacy code compatibility |

### Installation
```bash
pip install httpx
```

## Architecture Patterns

### Pattern 1: FastAPI Exception Handlers

**app.py or main.py:**
```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from graph_rlm.backend.src.core.exceptions import BaseGraphRLMError, ValidationError

app = FastAPI()

@app.exception_handler(BaseGraphRLMError)
async def graphrlm_exception_handler(request: Request, exc: BaseGraphRLMError):
    """Handle all GraphRLM exceptions with proper HTTP status codes."""
    status_code = exc.http_status_code if hasattr(exc, 'http_status_code') else 500
    return JSONResponse(
        status_code=status_code,
        content=exc.to_dict()
    )

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors with 422 status code."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": str(exc),
            "details": exc.details if hasattr(exc, 'details') else None
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Pass through FastAPI HTTPExceptions."""
    raise exc
```

**Exception base class enhancement:**
```python
class BaseGraphRLMError(Exception):
    """Base exception with HTTP status code support."""
    
    def __init__(
        self,
        message: str,
        correlation_id: str = None,
        http_status_code: int = 500,
        **kwargs
    ):
        self.message = message
        self.correlation_id = correlation_id
        self.http_status_code = http_status_code
        self.kwargs = kwargs
        super().__init__(message)

    def to_dict(self):
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "correlation_id": self.correlation_id,
            **(self.kwargs or {})
        }
```

### Pattern 2: HTTP Status Code Mapping

| Exception Type | Status Code | When Raised |
|----------------|-------------|-------------|
| ValidationError | 422 | Input validation fails |
| ExternalServiceError | 503 | External service unavailable |
| GraphError | 500 | Graph database errors |
| CircuitOpenError | 503 | Circuit breaker open |
| BaseGraphRLMError | 500 | Default |

### Pattern 3: MCP Server Circuit Breaker Validation

```python
# tests/integration/test_mcp_circuit_breaker.py
import pytest
from graph_rlm.backend.src.core.services.circuit import CircuitBreaker, CircuitOpenError
from graph_rlm.backend.src.mcp_integration import safe_mcp_call

@pytest.mark.asyncio
async def test_mcp_circuit_open_behavior(mock_registry):
    """Verify MCP calls fail gracefully when circuit opens."""
    circuit = CircuitBreaker(
        failure_threshold=3,
        timeout=30.0,
        success_threshold=2,
        name="mcp_server"
    )
    
    # Simulate circuit opening
    for _ in range(3):
        with pytest.raises(ExternalServiceError):
            await circuit.call_async(safe_mcp_call, "unavailable_server")
    
    # Circuit should now be open
    with pytest.raises(CircuitOpenError) as exc_info:
        await circuit.call_async(safe_mcp_call, "unavailable_server")
    
    assert exc_info.value.circuit_name == "mcp_server"
```

### Pattern 4: pytest-cov Incremental Coverage

**pyproject.toml update:**
```toml
[tool.pytest.ini_options]
addopts = """
    -v
    --tb=short
    --cov=src
    --cov-context=test
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-report=lcov:coverage.lcov
"""

[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/__pycache__/*", "*/migrations/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
]
```

**Coverage reporting command:**
```bash
# Full coverage report
pytest --cov=src --cov-report=term-missing --cov-report=html

# Incremental coverage (compare to previous)
pytest --cov=src --cov-report=lcov --cov-report=term-missing
lcov --diff coverage.lcov previous.lcov 2>/dev/null || diff-cover coverage.lcov --diff=previous.lcov
```

### Pattern 5: Full Test Suite Validation

```python
# tests/conftest.py additions
@pytest.fixture(scope="session")
def full_test_run():
    """Validate full test suite can execute."""
    import subprocess
    result = subprocess.run(
        ["pytest", "--collect-only", "-q"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Test collection failed: {result.stderr}"
    test_count = int(result.stdout.strip().split()[-1]) if result.stdout else 0
    return test_count

def test_full_suite_passes():
    """Verify all tests pass in one run."""
    import subprocess
    result = subprocess.run(
        ["pytest", "-x", "--tb=short"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Tests failed: {result.stderr}"
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HTTP exception handling | Custom middleware | FastAPI add_exception_handler | Built-in, handles all edge cases |
| Status code mapping | Dictionary lookup | Exception class attribute | Type-safe, extensible |
| Coverage reporting | Custom scripts | pytest-cov | Accurate, CI-integrated |
| API testing | requests library | httpx TestClient | FastAPI native, async support |

**Key insight:** FastAPI's exception handling system is designed for exactly this use case. Adding custom exception handlers at the app level is the standard pattern.

## Common Pitfalls

### Pitfall 1: Exception Handler Order
**What goes wrong:** Generic handler catches specific exceptions before specialized handlers
**Why it happens:** FastAPI uses first-matching handler
**How to avoid:** Register specific handlers (ValidationError) before general handlers (BaseGraphRLMError)
**Warning signs:** 500 errors instead of 422 for validation

### Pitfall 2: Circular Imports
**What goes wrong:** ImportError when importing app in test modules
**Why it happens:** app imports exceptions, tests import app
**How to avoid:** Use lazy imports or create separate exception handlers module
**Warning signs:** "cannot import name 'app' from 'main'"

### Pitfall 3: Coverage Gaps in API Layer
**What goes wrong:** API endpoints not covered by tests
**Why it happens:** Only unit tests exist, no integration tests
**How to avoid:** Add endpoint tests with TestClient
**Warning signs:** API module shows 0% coverage

### Pitfall 4: Incomplete Coverage Reports
**What goes wrong:** Coverage reports don't show missing lines
**Why it happens:** Not running with --cov-context=test or missing data collection
**How to avoid:** Use --cov-context=test for detailed reports
**Warning signs:** "Coverage warning: No data to report"

## Code Examples

### API Endpoint with Exception Handling

```python
# src/api/endpoints/agent.py
from fastapi import APIRouter, HTTPException
from graph_rlm.backend.src.core.agent import Agent
from graph_rlm.backend.src.core.exceptions import ValidationError

router = APIRouter()
agent = Agent()

@router.post("/agent/run")
async def run_agent(prompt: str):
    """Run agent with prompt."""
    try:
        # Validate input
        if len(prompt) < 10:
            raise ValidationError(
                message="Prompt too short",
                error_code="VALIDATION_PROMPT_TOO_SHORT",
                http_status_code=422
            )
        
        # Run agent
        result = await agent.main_loop(prompt)
        return {"result": result}
        
    except ValidationError:
        raise  # Let FastAPI handler convert to 422
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Test Client Setup

```python
# tests/api/conftest.py
import pytest
from fastapi.testclient import TestClient
from graph_rlm.backend.main import app

@pytest.fixture
def test_client():
    """Create test client for API testing."""
    return TestClient(app)

def test_endpoint_returns_validation_error(test_client):
    """Test that short prompts return 422."""
    response = test_client.post("/agent/run", json={"prompt": "hi"})
    assert response.status_code == 422
```

### Coverage Report Script

```bash
#!/bin/bash
# scripts/coverage-report.sh
echo "=== Running Full Test Suite ==="
pytest -x --tb=short -q

echo ""
echo "=== Generating Coverage Report ==="
pytest --cov=src --cov-report=term-missing --cov-report=html

echo ""
echo "=== Coverage Summary ==="
coverage report --show-missing
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual HTTPException raise | Exception handlers | FastAPI 0.100+ | Cleaner code, automatic status codes |
| coverage.py standalone | pytest-cov | pytest-cov 4.0+ | Unified workflow |
| requests for testing | httpx TestClient | FastAPI async | Native async support |

**Deprecated/outdated:**
- Manual status code mapping in each endpoint
- coverage.py without pytest-cov integration
- Synchronous test clients for async endpoints

## Open Questions

1. **FastAPI Version Compatibility**
   - What we know: FastAPI >=0.128.0 in dependencies
   - What's unclear: Specific exception handler API changes
   - Recommendation: Test with actual FastAPI version installed

2. **MCP Server Test Environment**
   - What we know: MCP integration exists from Phase 2
   - What's unclear: Can we run MCP server in tests or need full mocks?
   - Recommendation: Use mock MCP server for circuit breaker tests

## Sources

### Primary (HIGH confidence)
- FastAPI official documentation - exception handling patterns
- pytest-cov documentation - coverage configuration
- Current codebase - Phase 1-4 implementations

### Secondary (MEDIUM confidence)
- FastAPI exception handling tutorials - community patterns
- pytest-cov incremental coverage - CI pipeline examples

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - well-documented, stable libraries
- Architecture: HIGH - standard FastAPI patterns
- Pitfalls: MEDIUM - version-specific nuances may exist

**Research date:** 2026-02-12
**Valid until:** 2026-03-12 (30 days)