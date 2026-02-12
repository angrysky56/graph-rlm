# Graph-RLM: Recursive Language Model for Developers

## What This Is

A Graph-based Recursive Language Model (RLM) system that enables developers to process context windows 2 orders of magnitude larger than model limits. The system uses externalized context management, programmatic filtering, and recursive decomposition to query infinite context efficiently without degrading reasoning performance.

## Core Value

Developers can work with context (codebases, documentation, research) far exceeding model limits while maintaining cost efficiency and reasoning quality through inference-time scaling rather than massive training runs.

## Requirements

### Validated

(None yet — engineering health improvements pending validation)

### Active

- [ ] **EH-01**: Replace 141 broad exception handlers with specific exception types + logging
- [ ] **EH-02**: Implement custom exception hierarchy with error codes
- [ ] **EH-03**: Create comprehensive unit test infrastructure (pytest, fixtures, mocking)
- [ ] **EH-04**: Add test coverage for critical paths (agent main loop, recursive reasoning)
- [ ] **EH-05**: Add test coverage for error scenarios (MCP disconnects, DB failures, LLM timeouts)
- [ ] **EH-06**: Implement circuit breaker pattern for LLM APIs and MCP servers
- [ ] **EH-07**: Remove duplicate LLM_PROVIDER config definition
- [ ] **EH-08**: Refactor fragile modules for testability (skill storage, MCP client, axiom system)
- [ ] **EH-09**: Upgrade LangChain 1.x to 2.x (or pin and document upgrade path)
- [ ] **EH-10**: Pin MCP server version and verify compatibility
- [ ] **EH-11**: Add centralized error logging and alerting framework
- [ ] **EH-12**: Implement proper connection handling tests for MCP client

### Out of Scope

- **Security hardening** — Home lab use case, .env gitignored, keys user-managed
- **Large file refactoring** — Unless required for testability (agent.py, dream.py, desktop_commander.py)
- **Graph cleanup/TTL policies** — Defer to post-health phase
- **Redis session store** — Defer to post-health phase
- **Rate limiting implementation** — Defer to post-health phase (LLM API backoff)

## Context

### RLM Architecture Overview

The system achieves inference-time scaling through five efficiency mechanisms:

1. **Externalized Context Management**: Long prompts stored as string variables in persistent Python REPL rather than loaded into model context. Model interacts symbolically (e.g., `len(prompt)`, `prompt[:100]`) to decouple data size from context limits.

2. **Programmatic Filtering & Peeking**: Code-based filtering using string manipulation and regex to "peek" into context without neural network processing. Leverages model priors to generate targeted queries, discarding irrelevant sections at zero token cost.

3. **Recursive Decomposition**: The `llm_query` function enables sub-calls for complex tasks. Divides text by chapters/sections, runs isolated reasoning chains to prevent "context rot."

4. **Persistent State & Output Stitching**: Uses REPL memory as variable buffer. Intermediate results saved to variables, aggregated programmatically or via final summarizing sub-call.

5. **Cost Efficiency**: Selective viewing instead of full-context ingestion. Batch processing (e.g., 200k characters per sub-call) minimizes API call overhead.

### Existing Codebase State

- **Codebase Map**: Mapped at `.planning/codebase/` (7 documents)
- **Tech Stack**: Python 3.13+, LangChain 1.x, MCP 1.26, FalkorDB graph
- **Critical Files**: agent.py (2,292 lines), dream.py (1,135 lines), desktop_commander.py (1,696 lines)
- **Skills System**: 60+ skills in `skills/` directory with MCP integration
- **Axioms**: 10+ disabled axioms in `_disabled/` subdirectory

### Known Issues from Codebase Analysis

- 141 instances of `except Exception` or bare `except` blocks silently swallowing errors
- Multiple `except RuntimeError: pass` and `except asyncio.CancelledError: pass` statements
- No custom exception hierarchy
- No circuit breaker for external services
- No unit tests (only verification scripts in `tests/`)
- Duplicate `LLM_PROVIDER` config definition

## Constraints

- **Tech Stack**: Python 3.13+, LangChain, MCP, FalkorDB — maintain compatibility
- **Use Case**: Home lab/HPC environment, not multi-tenant production
- **Security**: Low priority — .env gitignored, user-managed keys, no external access
- **Performance**: Maintain inference-time scaling efficiency; circuit breaker should not add significant latency
- **Testing**: All fixes must include tests; no "works on my machine" acceptance

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Address all engineering health issues before new features | Foundation must be reliable for recursive reasoning to function correctly | — Pending |
| Skip security hardening | Home lab context, .env gitignored, user-managed keys | — Pending |
| Upgrade LangChain to 2.x | LangChain 1.x in maintenance mode, security patches eventual stop | — Pending |
| Add circuit breaker for external services | Prevent cascade failures from LLM/MCP outages | — Pending |
| Create custom exception hierarchy | Replace broad handlers, enable proper error propagation | — Pending |

---

*Last updated: 2026-02-12 after CONCERNS.md analysis*