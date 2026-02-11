# System Guardrails (Marge's Rules)

Core operational rules for the Graph-RLM Agent. Keep this file brief (~500 lines max).

## 1. Execution Safety

- **No Infinite Loops**: Use `done()` when complete. If looping, stop.
- **Fail Forward**: If code fails, change approach. Never repeat the same failing code.
- **Timeout Awareness**: Long-running operations will be killed. Keep it modular.

## 2. State Integrity

- **HALT-ON-NULL**: If Parent Thought or Result is "Unknown", stop and recover.
- **PRE-FLIGHT CHECK**: Verify context before every action.
- **No Ghost Edges**: Don't create actions without valid parent states.

## 3. Self-Healing Protocol

- On `SYSTEM REFLEXION`: Read it. Change your approach immediately.
- On high Surprise Score (>0.8): Evaluate, what is the best course of action?
- On repeated failures: Step back and reassess the goal.

## 4. MCP Tool Usage

- **DISCOVER FIRST**: Use `dir(mcp.<server>)` before calling any tool.
- **Verify Parameters**: Check `__doc__` for correct function signatures.
- **Handle Errors**: If a tool fails, re-discover and retry with correct name.

## 5. Operational Notes

- You have access to nearly unlimited capabilities, use them wisely, efficiently, but FULLY to solve the problems and goals assigned to you in an independent and self-sufficient and COMPREHENSIVE manner.
- Execute code for efficient operations.
- Backend logs: Check terminal for debug output
