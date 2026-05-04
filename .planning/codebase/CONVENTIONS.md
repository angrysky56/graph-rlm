# Code Conventions

## Overview

This document describes the coding conventions and standards used in the Graph-RLM project. The codebase follows modern Python best practices with a focus on:
- Type safety and clear interfaces
- Async/await patterns throughout
- Comprehensive error handling
- Clear documentation and naming

---

## Code Style and Formatting

### Python Version
- **Minimum:** Python 3.13
- Uses modern syntax features (structural pattern matching, type unions, etc.)

### Import Organization
Imports should be organized in the following order:

1. Standard library imports (datetime, pathlib, typing, etc.)
2. Third-party imports (httpx, numpy, etc.) - grouped alphabetically
3. Local application imports (relative or absolute)

### Indentation and Whitespace
- **Use 4 spaces** per indentation level (no tabs)
- Lines should be max **100 characters**
- Double quotes preferred for strings
- No trailing whitespace on any line
- Blank lines between logical sections

### Docstrings

#### Module Docstrings
```python
"""
Module-level docstring.

Brief description of what the module does, including:
- Purpose and functionality
- Key components/classes
- Usage examples (if applicable)
"""
```

#### Class Docstrings
```python
class ClassName:
    """
    Class-level description.

    Detailed explanation of the class's purpose and responsibilities.
    
    Attributes:
        attribute_name (type): Description of the attribute
    
    Notes:
        Additional context or implementation details.
    """
```

#### Function/Method Docstrings
```python
def method_name(self, param1: Type, param2: Optional[Type]) -> ReturnType:
    """
    Brief, concise description of what the method does.

    Extended description of functionality:
    - Parameter explanations
    - Return value description
    - Exceptions raised
    - Usage examples

    Args:
        param1 (Type): Description of param1
        param2 (Optional[Type]): Description of param2, can be None

    Returns:
        ReturnType: Description of return value

    Raises:
        CustomException: When X happens

    Examples:
        >>> method_name("arg1", "arg2")
        "result"
    """
```

#### Private Method Documentation
```python
def _private_method(self, param: Type) -> ReturnType:
    """Implementation details. Use public wrapper instead."""
```

### Type Annotations

Use explicit type hints for all function parameters and return values:

```python
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

def process_data(data: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
    """Process data with type safety."""
    result: List[Dict[str, Any]] = []
    for item in data.values():
        result.append(item)
    return result
```

#### Special Cases:
- Use `Optional[T]` instead of `Union[T, None]`
- Use `Any` for complex/unknown types
- Use `TYPE_CHECKING` imports only when checking types at runtime

---

## Naming Conventions

### Files and Directories

#### Module Files
- Use **snake_case**: `my_module.py`
- Single word names preferred for small modules
- Verb-noun for functional modules: `query_repl.py`, `execute_code.py`

#### Classes
- Use **PascalCase**: `Agent`, `Dreamer`, `PythonREPL`
- Use descriptive, noun-based names
- Single word names acceptable for utility classes

#### Functions and Methods
- Use **snake_case**
- Public methods: `method_name()`
- Private methods: `_method_name()` (single underscore)
- Protected methods: `__method_name__()` (double underscore for name mangling)

#### Variables and Constants

- Constants (module-level): `MAX_RECURSION_DEPTH = 3`
- Instance variables (snake_case): `self.session_id = None`
- Private instance variables: `self._internal_state = None`
- Class constants: `MY_CLASS_CONSTANT = 42`
- Class-level variables: `MY_CLASS_CONSTANT = 42`

### Async Methods

- Methods that are `async def` should end in `_async` or be clearly marked as async
- Public async methods: `async def process()`
- Private async helpers: `async def _process_async()`

### Boolean Variables

- Prefer `is_` prefix for boolean variables
  - `is_valid = True`
  - `should_stop = False`
  - `has_permission = False`

---

## Code Organization Patterns

### Module Structure

A typical module should follow this structure:

1. **Module docstring** at the top
2. **Standard library imports** (ordered)
3. **Third-party imports** (ordered alphabetically)
4. **Local imports** (relative or absolute)
5. **Constants and configuration**
6. **Class and function definitions**
7. **Example usage** (in `if __name__ == "__main__":` block)

### Class Organization

```python
class MyClass:
    """Class-level docstring."""
    
    # Class constants
    CLASS_CONSTANT = 42
    
    # Class methods
    @classmethod
    def create_instance(cls, data: Dict) -> "MyClass":
        """Create instance from data."""
        return cls()
    
    @staticmethod
    def static_helper(value: int) -> int:
        """Static helper method."""
        return value * 2
    
    # Private class methods
    @classmethod
    def _validate_input(cls, data: Dict) -> bool:
        """Internal validation."""
        return True
    
    def __init__(self, param: Type):
        """Initialize the instance."""
        self.instance_variable = param
    
    # Public methods
    def public_method(self) -> ReturnType:
        """Public method description."""
        return self._private_method()
    
    # Private methods
    def _private_method(self) -> ReturnType:
        """Private method description."""
        return None
```

### Function Organization

```python
def public_function(param1: Type1, param2: Type2) -> ReturnType:
    """Function-level docstring."""
    # Constants
    MAX_VALUE = 100
    
    # Validation
    if not validate_input(param1):
        raise ValueError("Invalid input")
    
    # Processing
    result = process_data(param1, param2)
    
    return result
```

### Method Organization Within Classes

Methods should be organized in this order:
1. `__init__` and setup methods
2. Public API methods
3. Private helper methods
4. Special methods (`__str__`, `__repr__`, etc.)

---

#---

## Error Handling Patterns

### Structured Exception Hierarchy

The project uses a centralized exception hierarchy defined in `graph_rlm.backend.src.core.exceptions`. All custom exceptions must inherit from `BaseGraphRLMError`.

```python
class BaseGraphRLMError(Exception):
    """Base exception for all Graph-RLM errors."""
    def __init__(self, message: str, error_code: ErrorCode, http_status_code: int = 500):
        self.error_code = error_code
        self.http_status_code = http_status_code
        super().__init__(message)
```

**Common Exception Types:**
- `CoreError`: For fundamental system failures.
- `GraphError`: For issues interacting with FalkorDB.
- `SkillExecutionError`: For failures within skills.
- `ExternalServiceError`: For LLM or API timeout/failures (triggers circuit breakers).
- `ValidationError`: For input or business rule violations.

### Circuit Breaker Pattern

Critical external calls (LLM, MCP) should be wrapped in a circuit breaker to prevent cascade failures.

```python
from graph_rlm.backend.src.core.circuit import CircuitBreaker

# Example usage
circuit = CircuitBreaker(name="llm_service", config=my_config)

async def call_llm(prompt: str):
    async with circuit:
        return await llm_service.ainvoke(prompt)
```

### Logging with Context

Use `structlog` for structured logging. Always include relevant context like `session_id` and `correlation_id`.

```python
from graph_rlm.backend.src.core.logging import get_logger

logger = get_logger(__name__)

def my_function(session_id: str):
    log = logger.bind(session_id=session_id)
    log.info("Processing task", task_type="reasoning")
```

---

## Testing Documentation

### pytest Infrastructure

The project uses **pytest** with **pytest-asyncio**. Reusable mocks are managed via a centralized `MockRegistry`.

#### MockRegistry Usage

Use the `mock_registry` fixture to access pre-configured mocks for common dependencies.

```python
@pytest.mark.asyncio
async def test_agent_query(mock_registry):
    # Arrange
    mock_llm = mock_registry.llm
    mock_llm.ainvoke.return_value = MagicMock(content="Mocked answer")
    
    agent = Agent()
    agent.llm = mock_llm
    
    # Act
    result = await agent.query("Hello")
    
    # Assert
    assert "Mocked answer" in result
```

---

## Documentation Standards

### Module Documentation

Every module should have a comprehensive docstring:

```python
"""
Module Name

Brief 1-2 sentence summary of what this module does.

Detailed description covering:
- Purpose and functionality
- Key components and classes
- Usage patterns
- Dependencies
- Example usage (if complex)
"""
```

### Class Documentation

Classes must have docstrings with:
1. **Summary sentence** (what it is)
2. **Detailed description** (how it works, what it does)
3. **Attributes** (key instance variables)

### Function Documentation

Functions require:
1. **One-line summary**
2. **Detailed description**
3. **Args section** (each parameter with type and description)
4. **Returns section** (description and type)
5. **Raises section** (if applicable)

### In-Line Comments

- Use **#** for inline comments
- Comments should be clear and concise
- Comment code that's not self-explanatory
- Prefer comments that explain *why* rather than *what*

### Docstring Conventions

#### Summary Line
- First line should be a summary sentence
- Use present tense ("Process data" not "Processes data")
- Start with a verb if describing a function

#### Parameters
- Format: `param_name (type): description`
- Include types in parentheses
- List all parameters even if optional

---

## Code Review Checklist

Before committing code, ensure:

- [ ] Follows naming conventions (snake_case for functions, PascalCase for classes)
- [ ] Has comprehensive docstrings (modules, classes, public methods)
- [ ] Uses type annotations for all parameters and returns
- [ ] Proper error handling with logging
- [ ] All imports are necessary
- [ ] No TODO comments without resolution
- [ ] No print statements (use logging)

---

*Last updated: 2026-05-04*
