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

## Error Handling Patterns

### General Principles

1. **Use specific exception types** when possible
2. **Handle errors at the right level** (don't swallow errors at higher levels)
3. **Log errors appropriately** (use logger instead of print)
4. **Return meaningful error states** rather than raising exceptions for control flow
5. **Use type annotations** to document expected exceptions

### Exception Handling

```python
from typing import Optional, Dict, Any
import logging

logger = get_logger(__name__)

def process_data(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process data with proper error handling."""
    try:
        # Input validation
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        
        # Processing
        result = _internal_processing(data)
        
        # Validation of output
        if not result:
            logger.warning("Processing returned None")
            return None
        
        return result
        
    except ValueError as e:
        # Log and re-raise specific errors
        logger.error("Input validation failed: %s", e)
        raise
    except Exception as e:  # pylint: disable=broad-except
        # Catch-all for unexpected errors
        logger.exception("Unexpected error in processing: %s", e)
        return None

def _internal_processing(data: Dict[str, Any]) -> Dict[str, Any]:
    """Internal processing logic."""
    # Implementation
    return data
```

### Async Error Handling

```python
async def async_operation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle async operation with proper error handling."""
    try:
        result = await _async_processing(data)
        return result
    except (ValueError, TypeError) as e:
        logger.error("Async validation failed: %s", e)
        raise
    except asyncio.TimeoutError:
        logger.error("Operation timed out")
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Unexpected async error: %s", e)
        raise
```

### Error Logging

```python
import logging

logger = get_logger(__name__)

# Error logging patterns

# 1. Simple error logging
try:
    result = risky_operation()
except ValueError as e:
    logger.error("Value error: %s", e)

# 2. Exception logging (with traceback)
try:
    result = risky_operation()
except Exception as e:
    logger.exception("Unexpected error: %s", e)

# 3. Warning logging (recoverable errors)
if not result:
    logger.warning("Operation returned empty result")

# 4. Debug logging (development)
if DEBUG_MODE:
    logger.debug("Debug info: %s", data)
```

### Error Return Patterns

```python
# Pattern 1: Return None for errors
def safe_operation() -> Optional[Dict]:
    """Return None on error."""
    try:
        result = do_operation()
        return result
    except Exception:
        return None

# Pattern 2: Return error codes/flags
def check_valid() -> bool:
    """Returns False on error."""
    try:
        # Validation logic
        return True
    except Exception:
        return False

# Pattern 3: Raise exceptions for critical errors
def critical_operation() -> Dict:
    """Raises exception on error."""
    try:
        result = do_operation()
        if not result:
            raise ValueError("Operation failed")
        return result
    except Exception as e:
        logger.error("Operation failed: %s", e)
        raise
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

Typical usage:
    >>> from module_name import ClassName
    >>> instance = ClassName()
    >>> result = instance.method()
"""

```

### Class Documentation

Classes must have docstrings with:
1. **Summary sentence** (what it is)
2. **Detailed description** (how it works, what it does)
3. **Attributes** (key instance variables)
4. **Usage notes** (important behaviors or edge cases)

```python
class Agent:
    """
    The core Agent class for Graph-RLM.

    This agent implements the Recursive Logic Machine (RLM) architecture,
    providing a persistent REPL and graph-based memory system.

    Key Features:
    - Persistent state across recursive calls
    - Graph-based memory using FalkorDB
    - Axiomatic consistency checking
    - Dream sleep cycles for wisdom consolidation

    Attributes:
        db: Graph database client for persistent storage
        llm: Language model service for reasoning and generation
        repl_manager: Manager for isolated REPL environments

    Examples:
        >>> agent = Agent()
        >>> result = await agent.query("Task description")
        >>> print(result)
    """
```

### Function Documentation

Functions require:
1. **One-line summary**
2. **Detailed description**
3. **Args section** (each parameter with type and description)
4. **Returns section** (description and type)
5. **Raises section** (if applicable)
6. **Examples** (for complex functions)

```python
def process_request(
    data: Dict[str, Any],
    timeout: float = 10.0,
    retry_attempts: int = 3
) -> Dict[str, Any]:
    """
    Process an incoming request with retry logic.

    This function handles incoming requests with built-in retry
    capabilities for transient failures. It validates input, processes
    the request through configured pipelines, and returns structured
    responses.

    Args:
        data: Input request data dictionary
        timeout: Maximum time to wait for processing in seconds
        retry_attempts: Number of retry attempts on failure

    Returns:
        Dict containing processing results with keys:
            - 'status': 'success' or 'error'
            - 'data': Processed data or error details
            - 'timestamp': Processing completion time

    Raises:
        ValueError: If input data is invalid
        TimeoutError: If operation exceeds timeout limit

    Examples:
        >>> result = process_request({"key": "value"})
        >>> assert result['status'] == 'success'
        
        >>> result = process_request(None)
        Traceback (most recent call last):
            ...
        ValueError: Input data cannot be None
    """
```

### In-Line Comments

- Use **#** for inline comments
- Comments should be clear and concise
- Comment code that's not self-explanatory
- Prefer comments that explain *why* rather than *what*

```python
# Good comments
# Store the result for later use in the analysis pipeline
# Use absolute path to avoid environment confusion
# Calculate the deviation from the target metric

# Avoid redundant comments
# result = data * 2  # Multiply by 2  <- What does this mean?

# Good inline comments
if result > threshold:  # Check if result exceeds safety threshold
    logger.warning("Result exceeds safety limit")
```

### Docstring Conventions

#### Summary Line
- First line should be a summary sentence
- Use present tense ("Process data" not "Processes data")
- Start with a verb if describing a function

#### Parameters
- Format: `param_name (type): description`
- Include types in parentheses
- Use `Optional[type]` for nullable parameters
- List all parameters even if optional

#### Returns
- Format: `type: description`
- Describe what the function returns
- Note any transformation of input data

#### Raises
- List exceptions that the function can raise
- Include `TypeError` for incorrect argument types
- Include `ValueError` for invalid argument values

#### Notes Section
- Add `Notes:` section for implementation details
- Document edge cases and corner cases
- Include performance considerations

---

## Testing Documentation

### Test File Structure

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
from unittest.mock import MagicMock, AsyncMock

# Setup path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock imports before actual imports
mock_db = MagicMock()
sys.modules["falkordb"] = mock_db
sys.modules["graph_rlm.backend.src.core.db"] = mock_db

# Import classes to test
from graph_rlm.backend.src.core.agent import Agent


class TestClassName(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for ClassName.
    
    Test cases organized by functionality:
    - TC1: Happy path
    - TC2: Error conditions
    - TC3: Edge cases
    """

    async def asyncSetUp(self):
        """Set up test fixtures before each test."""
        self.instance = ClassName()
        # Reset mocks
        
    async def test_happy_path(self):
        """Test the happy path behavior."""
        # Arrange
        data = {"key": "value"}
        
        # Act
        result = self.instance.method(data)
        
        # Assert
        self.assertEqual(result['status'], 'success')
    
    async def test_error_handling(self):
        """Test error handling behavior."""
        # Arrange
        invalid_data = None
        
        # Act & Assert
        with self.assertRaises(ValueError):
            self.instance.method(invalid_data)
```

---

## Additional Guidelines

### Logging

```python
import logging

logger = get_logger(__name__)

# Use appropriate log levels
logger.debug("Debug information")
logger.info("Important information")
logger.warning("Warning message")
logger.error("Error occurred")
logger.critical("Critical failure")

# Use formatted strings
logger.info("Processing item %s with value %d", item_id, value)

# Log exceptions with traceback
try:
    operation()
except Exception as e:
    logger.exception("Operation failed: %s", e)
```

### Environment Configuration

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Environment variables
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    
    # Feature flags
    ENABLE_FEATURE_X: bool = False
    MAX_RECURSION_DEPTH: int = 3

settings = Settings()
```

### Type Aliases

```python
from typing import Dict, List, Any, TypeAlias

# Define type aliases for clarity
MessageDict: TypeAlias = Dict[str, Any]
ResponseList: TypeAlias = List[Dict[str, Any]]
OptionalData: TypeAlias = Optional[Dict[str, Any]]

def process_data(data: MessageDict) -> ResponseList:
    """Process message data."""
    return []
```

---

## Code Review Checklist

Before committing code, ensure:

- [ ] Follows naming conventions (snake_case for functions, PascalCase for classes)
- [ ] Has comprehensive docstrings (modules, classes, public methods)
- [ ] Uses type annotations for all parameters and returns
- [ ] Proper error handling with logging
- [ ] No hardcoded values (use constants or config)
- [ ] Async functions are properly awaited
- [ ] Imports are organized correctly
- [ ] No print statements (use logging)
- [ ] No TODO comments without resolution
- [ ] Follows PEP 8 style guidelines
- [ ] Code is well-commented where necessary
- [ ] No commented-out code in production files
- [ ] Security-sensitive data not hardcoded
- [ ] All imports are necessary

---

*Last updated: February 9, 2026*
