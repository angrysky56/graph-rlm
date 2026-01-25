# Architectural Evaluation
**Date**: 2026-01-24
**Scope**: `backend/src/core/agent.py`, `backend/src/core/llm.py`, `backend/src/core/endpoints.py`, `backend/src/core/core.py`

## Critical Risks (Must Fix)
*   [ ] **[Code Execution Reliability]**: The system relies entirely on the LLM echoing back code in ` ```python ` blocks to execute it. If the user provides raw code (as in the screenshot), and the LLM acknowledges it without "executing" it (i.e. just talking about it), no code runs. This makes the REPL feel "broken".
*   [ ] **[Concurrency/Async Safety]**: `agent.py` spawns a `threading.Thread` to run `query_sync`, which calls `llm.generate`. While this prevents blocking the asyncio loop, the use of `queue.Queue` (thread-safe) to bridge to `async for` generator in `endpoints.py` is somewhat fragile if the thread dies silently. The `queue.get_nowait()` loop with `sleep(0.01)` is a busy-wait-ish pattern.
*   [ ] **[Error Propagation]**: `llm.generate` catches generic exceptions and returns a string starting with "Error: ". If the LLM call fails, the `_extract_code` logic simply finds no code, and the error string is treated as "thought" text. The user might not realize a technical failure occurred.

## Improvements (Should Fix)
*   [ ] **[Explicit Code Mode]**: If the user's input looks like python code (e.g. starts with `import` or valid syntax), the system should optionally wrap it or hint the LLM to execute it.
*   [ ] **[Streaming]**: `llm.generate` is blocking. Real-time token streaming would improve UX significantly.
*   [ ] **[REPL Persistence]**: Verification of `PythonREPL` implementation details is needed to ensure `locals()` are actually persisted correctly across calls.

## Strategic Recommendations
1.  **Enforce Code Execution**: Modify the system prompt or the parsing logic to better handle direct code inputs or partial code blocks.
2.  **Robust REPL Wrapper**: Ensure `PythonREPL` captures `stdout` and `stderr` reliably, even for imports or syntax errors.
