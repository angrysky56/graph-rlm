# Trace Analysis: Pipeline Debugging

## Phase 1: Injection
We need to insert "trace bullets" (logs) at every critical junction to see where the flow stops.

### Target Locations:
1. `graph_rlm/backend/src/core/endpoints.py`:
    - Entry of `chat_completions`.
    - Inside `response_stream` generator loop.
    - Exception handlers.
2. `graph_rlm/backend/src/core/agent.py`:
    - Entry of `query_sync`.
    - Before `llm.generate`.
    - After `llm.generate`.
    - Inside `_extract_code`.
    - Inside `_execute_code`.
3. `graph_rlm/backend/src/core/llm.py`:
    - Entry of `generate`.
    - Response received.

## Phase 2: Execution
User will retry the prompt. We will watch the terminal.

## Phase 3: Diagnosis
- If logs stop at `llm.generate`, it's an upstream API issue or timeout.
- If logs show response but no code execution, it's a Prompt/Parsing issue.
- If logs show execution but no frontend update, it's an Event/Frontend issue.
