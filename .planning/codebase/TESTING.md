# Testing Standards

## Overview

This document describes the testing patterns and standards used in the Graph-RLM project. The codebase uses **pytest** as the primary testing framework with **unittest** for isolated async tests.

---

## Testing Framework

### Primary Framework: pytest

The project uses pytest for all testing with the following configuration:

```toml
# pyproject.toml
[project]
dependencies = [
    "pytest>=9.0.2",
]

[dependency-groups]
dev = [
    "pytest-asyncio>=1.3.0",
]
```

**pytest.ini** configuration:
```ini
[pytest]
asyncio_mode = auto
```

### Async Testing Support

The project uses **pytest-asyncio** for async test support:
- `asyncio_mode = auto` enables automatic event loop management
- Tests can use `async def` for async test methods
- Mocking works seamlessly with async functions

---

## Test Structure and Organization

### Directory Structure

```
tests/
├── test_*.py              # Individual test files
├── verify_*.py           # Verification scripts (main-based)
├── skills/               # Skill-specific tests (if any)
└── scripts/              # Test utilities and scripts
```

### File Naming Conventions

- **Test files**: `test_*.py` (e.g., `test_agent.py`, `test_dream.py`)
- **Verification scripts**: `verify_*.py` (e.g., `verify_mcp_counts.py`)
- **Test classes**: `Test*` (e.g., `TestAgent`, `TestDreamer`)

### Class Naming Conventions

- **Test classes**: `Test*` (e.g., `TestValidationProtocol`, `TestStatelessAgent`)
- **Method names**: `test_*` (e.g., `test_happy_path`, `test_error_handling`)
- **Async test methods**: `async def test_*` (e.g., `async def test_async_operation`)

---

## Test Organization Patterns

### Module-Level Setup

Test modules should include:

```python
"""
Test module for [module_under_test].

Tests cover:
- Basic functionality
- Edge cases
- Error conditions
- Integration with other components
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

# 1. Setup PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 2. Mock imports BEFORE importing the module
# This prevents errors during import of modules with complex dependencies
mock_db_module = MagicMock()
mock_db_module.db = MagicMock()
sys.modules["graph_rlm.backend.src.core.db"] = mock_db_module

# 3. Import the classes to test
from graph_rlm.backend.src.core.agent import Agent
from graph_rlm.backend.src.core.dream import Dreamer
```

### Test Class Structure

Test classes should follow this structure:

```python
class TestClassName(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for ClassName.
    
    Organized by functionality:
    - TC1: Happy path/normal operation
    - TC2: Error handling
    - TC3: Edge cases/边界情况
    """
    
    async def asyncSetUp(self):
        """Set up test fixtures before each test."""
        self.instance = ClassName()
        # Reset mocks between tests
        mock_sheaf_module.sheaf.reset_mock()
        mock_repe_module.repe.reset_mock()
    
    async def asyncTearDown(self):
        """Clean up after each test."""
        pass
    
    # ====== Happy Path Tests ======
    
    async def test_happy_path(self):
        """Test normal operation with valid inputs."""
        # Arrange (Given)
        data = {"key": "value"}
        
        # Act (When)
        result = self.instance.method(data)
        
        # Assert (Then)
        self.assertEqual(result['status'], 'success')
        self.assertIsNotNone(result['data'])
    
    # ====== Error Handling Tests ======
    
    async def test_error_handling_invalid_input(self):
        """Test handling of invalid input parameters."""
        # Arrange
        invalid_data = None
        
        # Act & Assert (When/Then)
        with self.assertRaises(ValueError):
            self.instance.method(invalid_data)
    
    async def test_error_handling_timeout(self):
        """Test handling of operation timeouts."""
        # Arrange
        with patch.object(self.instance, 'method', side_effect=TimeoutError()):
            # Act & Assert
            with self.assertRaises(TimeoutError):
                self.instance.method("data")
    
    # ====== Edge Case Tests ======
    
    async def test_edge_case_empty_input(self):
        """Test behavior with empty input."""
        # Arrange
        data = {}
        
        # Act
        result = self.instance.method(data)
        
        # Assert
        self.assertEqual(result['status'], 'expected_status')
```

### Test Method Naming Convention

- Use descriptive names that describe what is being tested
- Follow this pattern: `test_<what>_should_<behavior>`
- Include context when necessary: `test_<feature>_with_<scenario>`

**Examples:**
- ✅ `test_happy_path`
- ✅ `test_error_handling_invalid_input`
- ✅ `test_mcp_discovery_structure`
- ✅ `test_axiomatic_consistency_basic`
- ❌ `test1`
- ❌ `test_something`

---

## Mocking and Stubbing Approaches

### Mocking Before Imports

**CRITICAL**: Always mock dependencies BEFORE importing the module being tested:

```python
# 1. Create mocks
mock_db = MagicMock()
mock_llm = MagicMock()
mock_sheaf = MagicMock()

# 2. Replace in sys.modules BEFORE import
sys.modules["falkordb"] = mock_db
sys.modules["graph_rlm.backend.src.core.db"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.llm"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.sheaf"] = MagicMock()

# 3. Now import the module
from graph_rlm.backend.src.core.agent import Agent
```

### AsyncMock for Async Functions

Use `AsyncMock` for async function mocking:

```python
from unittest.mock import AsyncMock

# Setup
mock_sheaf = MagicMock()
mock_sheaf.sheaf.diagnose_trace = AsyncMock()
mock_sheaf.sheaf.diagnose_trace.return_value = {"status": "HEALTHY"}

# Usage in test
result = await self.dreamer.validate_response(candidate, context)
self.assertEqual(result["status"], "valid")
```

### Patching Specific Methods

Use `patch` for targeted mocking:

```python
from unittest.mock import patch

async def test_with_patched_method(self):
    """Test with specific method mocked."""
    # Patch only the method, not the entire class
    with patch.object(
        self.dreamer,
        "analyze_holonomy",
        return_value="Custom response"
    ):
        result = await self.dreamer.dream_cycle()
        self.assertEqual(result, "Custom response")
```

### Patching Imports

Use `patch.object` or `patch.dict` for module-level imports:

```python
async def test_with_patched_import(self):
    """Test with module import patched."""
    with patch('graph_rlm.backend.src.core.sheaf') as mock_sheaf:
        mock_sheaf.sheaf.diagnose_trace.return_value = {"status": "TEST"}
        result = await self.dreamer.validate_response("test", "context")
        self.assertEqual(result["status"], "valid")
```

### Fixture Patterns

Define reusable fixtures in `conftest.py` (if needed):

```python
# conftest.py
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_db():
    """Common mock for database."""
    return MagicMock()

@pytest.fixture
def sample_agent():
    """Fixture for Agent instance."""
    return Agent()
```

---

## Coverage Standards

### Coverage Requirements

- **Minimum Coverage**: 80% for production code
- **Critical Path Coverage**: 100% for core agent logic
- **Async Code**: Aim for 90%+ coverage for async methods

### Running Tests with Coverage

```bash
# Run all tests with coverage
pytest --cov=graph_rlm --cov-report=html --cov-report=term-missing

# Run only tests for specific module
pytest --cov=graph_rlm.backend.src.core.agent tests/test_agent.py --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=graph_rlm --cov-report=html
# View report at: htmlcov/index.html
```

### Coverage Goals by Module

- **Core Agent Logic** (agent.py): 90%+
- **Database Layer** (db.py): 80%+
- **LLM Service** (llm.py): 85%+
- **Dreamer** (dream.py): 80%+
- **Sheaf Monitor** (sheaf.py): 85%+

### What to Test

**Essential to Test:**
- Happy path scenarios (normal operation)
- Error conditions (invalid input, failures)
- Edge cases (empty inputs, boundary values)
- Async behavior (correct async/await usage)
- Mock interaction (MCP tools, external dependencies)

**Less Critical to Test:**
- Pure utility functions (already well-tested by library code)
- Static helpers
- Configuration loading (well-tested by type system)

---

## Test Writing Guidelines

### AAA Pattern (Arrange, Act, Assert)

```python
async def test_method_name(self):
    """
    Test that method_name performs correctly.
    """
    # ====== Arrange (Given) ======
    input_data = {"key": "value"}
    expected_result = {"status": "success"}
    
    # ====== Act (When) ======
    result = await self.instance.method(input_data)
    
    # ====== Assert (Then) ======
    self.assertEqual(result, expected_result)
    self.assertEqual(result['status'], 'success')
```

### Assert Patterns

```python
# Equality assertions
assert result == expected
self.assertEqual(result, expected)

# Type assertions
assert isinstance(result, dict)
self.assertIsInstance(result, dict)

# Boolean assertions
assert result is not None
self.assertIsNotNone(result)

# Truthy/falsy assertions
assert result
self.assertTrue(result)

# Contains assertions
assert "key" in result
self.assertIn("key", result)

# Exception assertions
with self.assertRaises(ValueError):
    self.instance.method(None)
```

### Mock Verification

```python
from unittest.mock import MagicMock, call

async def test_with_verification(self):
    """Test with mock verification."""
    # Setup
    mock_method = MagicMock()
    self.instance.method = mock_method
    
    # Act
    result = await self.instance.process("data")
    
    # Verify
    mock_method.assert_called_once_with("data")
    mock_method.assert_called_with("data")
    
    # Verify specific call
    mock_method.assert_any_call("wrong_data")
    mock_method.assert_not_called()
    
    # Verify call count
    mock_method.assert_called_once()
    mock_method.assert_called_times(3)
```

### Async Test Patterns

```python
import asyncio
from unittest.mock import AsyncMock

async def test_async_operation(self):
    """Test async operation."""
    # Setup async mock
    self.async_service = AsyncMock()
    self.async_service.process = AsyncMock(return_value="result")
    
    # Act
    result = await self.async_service.process("input")
    
    # Assert
    self.assertEqual(result, "result")
    self.async_service.process.assert_awaited_once_with("input")

async def test_async_with_exception(self):
    """Test async operation with exception."""
    # Setup async mock to raise exception
    self.async_service = AsyncMock()
    self.async_service.process = AsyncMock(
        side_effect=ValueError("Error message")
    )
    
    # Act & Assert
    with self.assertRaises(ValueError):
        await self.async_service.process("input")
```

---

## Test Examples from Codebase

### Example 1: Agent Validation Test

```python
import asyncio
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock dependencies
mock_db = MagicMock()
sys.modules["falkordb"] = mock_db
sys.modules["graph_rlm.backend.src.core.db"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.llm"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.sheaf"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.repe"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.omcd"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.reflexion"] = MagicMock()

from graph_rlm.backend.src.core.agent import Agent, RLMInterface
from graph_rlm.backend.src.core.dream import Dreamer

class TestValidationProtocol(unittest.IsolatedAsyncioTestCase):
    """Test Agent-Dreamer Validation Protocol."""

    async def asyncSetUp(self):
        """Set up test fixtures."""
        self.dreamer = Dreamer()
        # Reset mocks
        mock_sheaf = MagicMock()
        mock_sheaf.sheaf.diagnose_trace = MagicMock()
        mock_repe = MagicMock()
        mock_repe.repe.scan_thought = MagicMock()
        mock_omcd = MagicMock()
        mock_omcd.omcd.evaluate_step = MagicMock()

    async def test_happy_path_validation(self):
        """TC1: Valid response passes all checks."""
        candidate = "The answer is 42."
        context = "Did deep thought."

        # Setup mocks
        mock_sheaf = MagicMock()
        mock_sheaf.sheaf.diagnose_trace = MagicMock(return_value={
            "status": "HEALTHY",
            "energy": 0.1,
        })
        
        mock_repe = MagicMock()
        mock_repe.repe.scan_thought = MagicMock(return_value={
            "Shakiness": 0.1,
        })

        result = await self.dreamer.validate_response(candidate, context)

        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["event"], "RLM_VALIDATED_RESPONSE")

    async def test_repe_shakiness_rejection(self):
        """TC2: High Shakiness receives WAKE event."""
        candidate = "I assume I am confused."
        
        mock_repe = MagicMock()
        mock_repe.repe.scan_thought = MagicMock(return_value={
            "Shakiness": 0.8,
        })
        
        result = await self.dreamer.validate_response(candidate, "ctx")

        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["event"], "RLM_WAKE")
        self.assertIn("Verify assumptions", result["instruction"])
```

### Example 2: PythonREPL Output Test

```python
import asyncio
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph_rlm.backend.src.core.core import PythonREPL

async def test_repl_output_capture():
    """Test PythonREPL output capture for MCP tools."""
    print("--- Verifying PythonREPL Output Capture ---")
    repl = PythonREPL()

    # Mock MCP tools
    mcp_mock = MagicMock()
    mcp_mock.skill_return = MagicMock()
    mcp_mock.skill_print = MagicMock()
    mcp_mock.skill_return.return_value = "Skill Result: 42"

    repl.namespace["mcp"] = mcp_mock

    # Test skill that returns value
    code1 = "await mcp.skill_return()"
    stdout, stderr, result, is_err = await repl.execute(code1)

    print(f"Code: {code1}")
    print(f"Stdout: '{stdout}'")
    print(f"Result: '{result}'")

    assert result == "Skill Result: 42", f"Expected 'Skill Result: 42', got '{result}'"
    print("✅ PASS: Return value captured.")

    # Test skill that prints
    code2 = "await mcp.skill_print()"
    stdout, stderr, result, is_err = await repl.execute(code2)

    print(f"Code: {code2}")
    print(f"Stdout: '{stdout}'")

    assert "Skill Output: Hello World" in stdout
    print("✅ PASS: Print output captured.")
```

### Example 3: Axiomatic Consistency Test

```python
import asyncio
import os
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph_rlm.backend.src.core.sheaf import sheaf
from graph_rlm.backend.src.mcp_integration.skills import get_skills_manager

async def test_axiom_blocking():
    """Test axiomatic consistency blocking."""
    print("🚀 Starting Axiomatic Consistency Verification...")

    # Create a test axiom
    mgr = get_skills_manager()
    axiom_name = f"axiom_test_{uuid4().hex[:4]}"
    axiom_code = """
def validate_max_value(val):
    '''Constraint: Value must be less than 50.'''
    if isinstance(val, (int, float)) and val >= 50:
        return False
    return True
"""

    mgr.save_skill(
        name=axiom_name,
        code=axiom_code,
        description="Value must be less than 50.",
        tags=["axiom", "test"],
    )

    try:
        # Test violating code
        violating_code = "result = 100\nresult"
        diag = sheaf.check_axiomatic_consistency(violating_code, task_tags=["physics"])
        print(f"  -> Violating Code Status: {diag['status']}")

        assert diag["status"] == "AXIOMATIC_VIOLATION", \
            f"Expected AXIOMATIC_VIOLATION, got {diag['status']}"

        # Test valid code
        valid_code = "result = 10\nresult"
        diag = sheaf.check_axiomatic_consistency(valid_code, task_tags=["physics"])
        print(f"  -> Valid Code Status: {diag['status']}")

        assert diag["status"] == "HEALTHY", \
            f"Expected HEALTHY, got {diag['status']}"

        print("✅ Axiomatic consistency basics passed!")
        return True
    finally:
        # Cleanup
        skill_file = mgr.skills_dir / f"{axiom_name}.py"
        if skill_file.exists():
            skill_file.unlink()
```

---

## Continuous Integration Testing

### CI Configuration

The project uses pytest with CI integration:

```yaml
# .github/workflows/test.yml (example)
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.13'
      - run: pip install -e .
      - run: pytest --cov=graph_rlm --cov-report=xml
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_agent.py

# Run specific test class
pytest tests/test_agent.py::TestAgent

# Run specific test method
pytest tests/test_agent.py::TestAgent::test_happy_path

# Run tests with verbose output
pytest -v

# Run tests with coverage
pytest --cov=graph_rlm --cov-report=term-missing

# Run only async tests
pytest -k async

# Run tests in debug mode
pytest --pdb
```

---

## Test Maintenance

### Regular Testing Checklist

- [ ] Run full test suite before committing
- [ ] Check coverage hasn't dropped
- [ ] Update tests when fixing bugs
- [ ] Add tests for new features
- [ ] Review test files in code reviews

### Test Documentation

All test files should include:
1. Module-level docstring explaining test coverage
2. Class-level docstring describing test suite
3. Test method docstrings describing what is tested

### Common Test Anti-Patterns

❌ **Bad:**
```python
def test_foo():
    result = foo()  # No assertion, no context
    return result

def test_bar():
    foo()  # No assertion at all

def test_baz():
    # No docstring, no description
    pass
```

✅ **Good:**
```python
def test_foo_with_valid_input():
    """
    Test that foo() returns correct value with valid input.
    """
    result = foo("input")
    assert result == "expected_output"

def test_bar_with_invalid_input():
    """Test that bar() raises ValueError for invalid input."""
    with pytest.raises(ValueError):
        bar(None)
```

---

## Additional Resources

### Useful pytest Features

- **Parametrized tests**: Test multiple inputs with one test
- **Fixture reuse**: Create reusable test components
- **Markers**: Tag tests for selective execution
- **Forked mode**: Run tests in isolated processes

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("input,expected", [
    ("valid", "result"),
    ("another", "result"),
    ("third", "result"),
])
async def test_parametrized(input, expected):
    assert process(input) == expected
```

---

*Last updated: February 9, 2026*
