# Implementation Plan: Fix RLM REPL & Import Paths

## Phase 1: Preparation
- [x] Analyzed `agent.py` and `core.py`.
- [x] Identified that `mcp_tools` import is failing because of incorrect path usage by user, and potential lack of feedback if LLM doesn't generate code block.

## Phase 2: Implementation (Agent Fixes)
- [ ] **[Step 1]**: Modify `graph_rlm/backend/src/core/agent.py`:
    -   Add `graph_rlm/backend` (or specific `mcp_tools` parent) to `sys.path` so `import mcp_tools` works directly. This fixes the immediate usability hurdle.
    -   Add `logger.info(f"LLM Response: {response_text}")` to debug silent failures.
    -   Update `system_prompt` to be even more clearer about code block formatting.

## Phase 3: Verification
- [ ] Restart backend (auto-reload should handle it).
- [ ] Use `run_command` or ask User to retry the import: `import mcp_tools.arxiv_mcp_server`.
