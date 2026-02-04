# Implementation Plan: Prompt-Code Integrity Alignment

## Phase 1: Preparation
- [x] Audit system prompt against `RLMInterface` implementation.
- [x] Verify REPL namespace injection logic in `_execute_code`.

## Phase 2: Implementation
- [ ] **[Step 1]**: Update `agent.py` -> `_build_system_prompt`.
    - *Context*: Add `await` requirement. Update `done()`, `agent`, `graph_search` references to `await rlm.*`.
- [ ] **[Step 2]**: Refactor Ethics section in the prompt for brevity and alignment with user principles (SOLID/Zen).

## Phase 3: Verification
- [ ] Mock a REPL session and verify that calling `await rlm.help()` returns expected results.
- [ ] Verify that the agent correctly identifies successful completion using `await rlm.done()`.

## Phase 4: Work Log
- Documented in `.CODEAGENCY/work_log.md`.
