---
wave: 1
depends_on: []
autonomous: true
files_modified:
  - graph_rlm/backend/src/core/config.py (remove duplicate LLM_PROVIDER)
---

# Phase 1: Foundation - Config Cleanup

## Overview

Remove duplicate LLM_PROVIDER configuration definition from Settings class. This is a quick win that eliminates confusion and potential inconsistencies.

## Requirements Addressed

- REFR-01: Remove duplicate LLM_PROVIDER config definition in Settings class

## Implementation Order

1. Identify duplicate definitions
2. Remove duplicate at line 46
3. Verify no functionality changes
4. Add documentation comment

## Tasks

### 5.1 Remove duplicate LLM_PROVIDER definition

In graph_rlm/backend/src/core/config.py:

Current state (lines 38 and 46):
```python
# Line 38:
LLM_PROVIDER: str = "openrouter"

# Lines 40-46:
REPL_TIMEOUT: int = 3000

GRAPH_NAME: str = "rlm_graph"

# LLM Settings (Primary Provider)
LLM_PROVIDER: str = "openrouter"  # ollama, openrouter, lmstudio, openai
```

Action: Remove lines 45-46 (the comment and duplicate definition), keeping line 38 as the canonical definition.

### 5.2 Add documentation comment

Add a comment above line 38 to clarify this is the primary LLM_PROVIDER definition:

```python
# Primary LLM Provider Configuration
# Valid values: "openrouter", "ollama", "lmstudio", "openai"
LLM_PROVIDER: str = "openrouter"
```

### 5.3 Verify functionality

The Settings class already uses LLM_PROVIDER in:
- get_config_for_provider() - line 72
- get_llm_config() - line 103

Since both reference self.LLM_PROVIDER, removing the duplicate is safe.

## Verification Criteria

- [ ] Line 46 (duplicate LLM_PROVIDER) removed
- [ ] Line 38 remains as canonical LLM_PROVIDER definition
- [ ] Settings.get_config_for_provider() still works
- [ ] Settings.get_llm_config() still works
- [ ] No runtime errors from the change

## Must Haves (Goal-Backward Verification)

1. **REFR-01 satisfied**: Duplicate LLM_PROVIDER config definition removed from Settings class
2. Settings class has single LLM_PROVIDER definition at line 38
3. All functionality that depends on LLM_PROVIDER continues to work
