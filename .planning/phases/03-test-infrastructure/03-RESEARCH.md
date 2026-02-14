# Phase 3: Test Infrastructure - Research

**Researched:** 2026-02-12
**Domain:** pytest configuration, async testing, mock registry patterns
**Confidence:** HIGH

## Summary

Phase 3 establishes test infrastructure for Graph-RLM with pytest-asyncio for async testing, a mock registry for external dependencies (FalkorDB, LLM services), and initial unit tests achieving 100% coverage on config.py and exceptions/base.py. The project already has pytest and pytest-asyncio in dependencies; configuration focuses on asyncio_mode=auto for Python 3.13+, proper conftest.py setup, and pytest-cov integration.

Key findings: Use pytest-asyncio>=0.24.0 for full Python 3.13 async/await support. Mock FalkorDB using its Python driver's session interface. Mock LLM services by intercepting LangChain chat model calls. pytest-cov should run with --cov-context=test for incremental tracking.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | >=9.0.2 | Test runner | Python standard, actively maintained |
| pytest-asyncio | >=0.24.0 | Async test support | Required for Python 3.13+ async/await |
| pytest-cov | >=6.0.0 | Coverage reporting | Integrates with pytest, generates reports |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest-mock | >=3.14.0 | Mock fixtures | For simple mocks in tests |
| pytest-xdist | (optional) | Parallel execution | For faster test runs |

### Installation
```bash
pip install pytest pytest-asyncio pytest-cov pytest-mock
```

## Architecture Patterns

### Recommended Project Structure
```
tests/
├── conftest.py              # Global fixtures
├── mocking/
│   ├── __init__.py
│   ├── mocks.py            # MockRegistry class
│   ├── falkordb.py         # FalkorDB mocks
│   ├── llm.py              # LLM service mocks
│   └── external.py         # HTTP/API mocks
├── unit/
│   ├── __init__.py
│   ├── test_config.py      # 100% coverage target
│   └── test_exceptions/    # exceptions/base.py tests
│       ├── __init__.py
│       └── test_base.py
└── legacy/
    └── test_agent.py       # agent.py with legacy mocking
```

### Pattern 1: pytest-asyncio Configuration

**pyproject.toml:**
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
```

**conftest.py:**
```python
import pytest
import asyncio
from tests.mocking.mocks import MockRegistry

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_registry():
    """Provide a fresh mock registry for each test."""
    registry = MockRegistry()
    yield registry
    registry.reset()
```

### Pattern 2: MockRegistry Class

```python
class MockRegistry:
    """Centralized mock management for FalkorDB, LLM services, and external APIs."""

    def __init__(self):
        self._mocks = {}
        self._reset_history = []

    def register(self, name: str, mock):
        """Register a mock by name."""
        self._mocks[name] = mock

    def get(self, name: str):
        """Retrieve a mock by name."""
        return self._mocks.get(name)

    def reset(self):
        """Reset all mocks and clear history."""
        self._reset_history.clear()
        for mock in self._mocks.values():
            if hasattr(mock, 'reset'):
                mock.reset()

    @property
    def falkordb(self):
        """Get FalkorDB mock."""
        return self._mocks.get('falkordb')

    @property
    def llm(self):
        """Get LLM service mock."""
        return self._mocks.get('llm')

    @property
    def external(self):
        """Get external API mock."""
        return self._mocks.get('external')
```

### Pattern 3: FalkorDB Mock

```python
from unittest.mock import MagicMock, AsyncMock
import pytest

@pytest.fixture
def mock_falkordb():
    """Create a mock FalkorDB client."""
    mock_client = MagicMock()
    mock_client.session = MagicMock()
    mock_client.session.query = MagicMock(return_value=[])
    mock_client.close = MagicMock()
    return mock_client

@pytest.fixture
def mock_registry_with_falkordb(mock_registry, mock_falkordb):
    """Provide mock registry with FalkorDB pre-registered."""
    mock_registry.register('falkordb', mock_falkordb)
    return mock_registry
```

### Pattern 4: LLM Service Mock

```python
from unittest.mock import patch, MagicMock
from langchain_openai import ChatOpenAI

@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    mock_model = MagicMock(spec=ChatOpenAI)
    mock_model.ainvoke = AsyncMock(return_value=MagicMock(content="Mocked response"))
    mock_model.abatch = AsyncMock(return_value=[MagicMock(content="Response 1"),
                                                 MagicMock(content="Response 2")])
    return mock_model

def use_mock_llm(mock_registry, mock_llm_service):
    """Configure registry to use mock LLM."""
    mock_registry.register('llm', mock_llm_service)
    # Patch in LangChain module
    with patch('langchain_openai.ChatOpenAI', return_value=mock_llm_service):
        yield
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Async event loops | Custom event loop management | pytest-asyncio's event_loop fixture | Handles lifecycle, cleanup, and scope correctly |
| Coverage tracking | Custom coverage logic | pytest-cov | Accurate measurement, report generation, CI integration |
| Mock reset | Manual mock tracking | MockRegistry class | Centralized, consistent reset across tests |

**Key insight:** pytest-asyncio handles all the complexities of async test execution including event loop lifecycle, fixture scopes, and cleanup. Custom solutions introduce bugs around loop closure and scope management.

## Common Pitfalls

### Pitfall 1: asyncio_mode Auto Import Issues
**What goes wrong:** `ImportError: cannot import name 'asyncio_mode' from 'pytest_asyncio'`
**Why it happens:** Using old pytest-asyncio version (< 0.24.0) or incorrect configuration syntax
**How to avoid:** Use pytest-asyncio>=0.24.0 and configure via `[tool.pytest.ini_options]` in pyproject.toml, not pytest.ini
**Warning signs:** Import errors, event loop not running tests

### Pitfall 2: Fixture Scope Mismatches
**What goes wrong:** "Event loop is closed" errors or fixtures not sharing state
**Why it happens:** Session-scoped event_loop with function-scoped fixtures that need it
**How to avoid:** Use function-scoped event_loop fixture for async tests, or carefully manage scope boundaries
**Warning signs:** Event loop errors, fixture dependency conflicts

### Pitfall 3: Incomplete Mock Reset
**What goes wrong:** Tests pass in isolation but fail when run together
**Why it happens:** Mocks not properly reset between tests, state leakage
**How to avoid:** Use MockRegistry with proper reset() calls in fixture teardown
**Warning signs:** Order-dependent test failures, "called more times than expected" mock errors

### Pitfall 4: Coverage Gaps from Exception Handlers
**What goes wrong:** 100% coverage goal missed due to untested exception paths
**Why it happens:** Exception constructors and context preservation logic not exercised
**How to avoid:** Write explicit tests for exception initialization with various parameters, test to_dict() serialization
**Warning signs:** Coverage reports showing lines in exceptions/base.py as uncovered

## Code Examples

### Async Test Example
```python
import pytest

@pytest.mark.asyncio
async def test_async_operation(mock_registry):
    """Test an async operation with mocked dependencies."""
    result = await mock_registry.falkordb.session.query("MATCH (n) RETURN n")
    assert result == []
```

### Coverage Configuration
```toml
[tool.pytest.ini_options]
addopts = """
    -v
    --tb=short
    --cov=src.core.config
    --cov=src.core.exceptions.base
    --cov-context=test
    --cov-report=term-missing
    --cov-report=html:htmlcov
"""

[tool.coverage.run]
source = ["src.core"]
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| pytest.mark.asyncio decorator | asyncio_mode = "auto" | pytest-asyncio 0.21+ | Simpler test syntax, no decorator needed |
| Manual event loop management | Built-in event_loop fixture | pytest-asyncio 0.21+ | Proper lifecycle, no leaks |
| coverage.py standalone | pytest-cov integration | 2018+ | Unified workflow, report generation |

**Deprecated/outdated:**
- pytest-asyncio < 0.24.0: Lacks full Python 3.13 support
- pytest.ini configuration: pyproject.toml preferred (PEP 517/518)
- Manual mock tracking: Use pytest-mock or custom MockRegistry

## Open Questions

1. **FalkorDB Python Driver Specifics**
   - What we know: FalkorDB has a Python client with session-based queries
   - What's unclear: Exact async support in the driver, connection pooling behavior
   - Recommendation: Test FalkorDB mock with real driver interface patterns, adjust if async support differs

2. **agent.py Legacy Mocking Pattern**
   - What we know: REFR-02 requires "isolated test module with legacy mocking"
   - What's unclear: What "legacy mocking" specifically means for this codebase
   - Recommendation: Inspect src/core/agent.py for existing test patterns, match that style

## Sources

### Primary (HIGH confidence)
- pytest-asyncio documentation - asyncio_mode configuration
- pytest-cov documentation - coverage configuration options
- pyproject.toml - current project dependencies and Python version

### Secondary (MEDIUM confidence)
- pytest-mock documentation - mock fixture patterns
- LangChain testing patterns - community examples for LLM mocking

### Tertiary (LOW confidence)
- FalkorDB Python client patterns - may need verification with actual driver

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - pytest ecosystem is mature and well-documented
- Architecture: HIGH - standard pytest patterns, MockRegistry is straightforward
- Pitfalls: MEDIUM - Python 3.13 asyncio is newer, edge cases may exist

**Research date:** 2026-02-12
**Valid until:** 2026-03-12 (30 days - pytest ecosystem is stable)