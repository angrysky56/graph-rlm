---
wave: 2
depends_on: ["plan-01-exception-base"]
autonomous: true
files_modified:
  - graph_rlm/backend/src/core/exceptions/serialization.py (new)
  - graph_rlm/backend/src/core/exceptions/base.py (update with serialization)
---

# Phase 1: Foundation - Exception Serialization

## Overview

Implement exception-to-dict serialization for API error responses. This enables structured error formats for FastAPI responses and log aggregation systems.

## Requirements Addressed

- EXCP-06: Exception-to-dict serialization for API error responses

## Implementation Order

1. Add serialization methods to BaseGraphRLMError
2. Implement to_dict() with cause chain support
3. Add to_json() convenience method
4. Update __init__.py exports

## Tasks

### 3.1 Add serialization methods to BaseGraphRLMError (base.py)

Extend BaseGraphRLMError with:

```python
def to_dict(self, *, include_cause: bool = True, include_stacktrace: bool = False) -> dict[str, Any]:
    """Convert exception to dictionary for API responses.
    
    Args:
        include_cause: Include cause chain in output
        include_stacktrace: Include formatted stacktrace
    
    Returns:
        JSON-compatible dictionary representation
    """
    result: dict[str, Any] = {
        "error": {
            "code": self.error_code.value if self.error_code else None,
            "message": self.message,
            "category": (
                self.error_code.category 
                if self.error_code and hasattr(self.error_code, "category") 
                else None
            ),
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
        },
        "context": dict(self._context),
    }
    
    if include_cause and self.__cause__ is not None:
        if isinstance(self.__cause__, BaseGraphRLMError):
            result["cause"] = self.__cause__.to_dict(
                include_cause=True, 
                include_stacktrace=False
            )
        else:
            result["cause"] = {
                "error": {
                    "message": str(self.__cause__),
                    "type": type(self.__cause__).__name__,
                }
            }
    
    if include_stacktrace:
        result["stacktrace"] = self.format_traceback()
    
    return result

def to_json(self, **kwargs) -> str:
    """Convert exception to JSON string.
    
    Args:
        kwargs: Additional arguments for json.dumps
    
    Returns:
        JSON string representation
    """
    import json
    return json.dumps(self.to_dict(), **kwargs)

def format_traceback(self) -> str:
    """Format exception traceback as string."""
    import traceback
    return "".join(traceback.format_exception(
        type(self), self, self.__traceback__
    )).strip() if self.__traceback__ else ""
```

### 3.2 Update __init__.py exports

Add to __init__.py:
```python
__all__ = [
    # ... existing exports ...
]
```

No additional exports needed - methods are on exception classes.

## Verification Criteria

- [ ] error.to_dict() returns JSON-compatible dictionary
- [ ] Dictionary includes error.code, message, category, timestamp, correlation_id
- [ ] Dictionary includes context dict with all additional parameters
- [ ] Cause chain is recursively serialized for BaseGraphRLMError causes
- [ ] Non-Graph-RLM causes are serialized with message and type
- [ ] to_json() returns valid JSON string
- [ ] format_traceback() returns formatted traceback string

## Must Haves (Goal-Backward Verification)

1. **EXCP-06 satisfied**: Developer can call error.to_dict() on any exception to get serializable API response format
2. API responses include error code, message, category, timestamp, correlation_id
3. Cause chain is preserved in serialized output
