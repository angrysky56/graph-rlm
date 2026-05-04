# Codebase Concerns & Technical Debt

**Analysis Date:** 2026-05-04

## Tech Debt

### [DONE] Exception Handling is inconsistent
- **Status:** RESOLVED (Phase 5)
- **Problem:** Broad `except Exception:` and missing specific exception types.
- **Resolution:** Implemented a structured exception hierarchy in `core/exceptions.py`. Replaced broad catches with specific types (`GraphError`, `ExternalServiceError`, etc.) in critical paths.

### [DONE] No standard unit testing framework
- **Status:** RESOLVED (Phase 3)
- **Problem:** Ad-hoc verification scripts rather than a proper test suite.
- **Resolution:** Adopted `pytest` + `pytest-asyncio`. Created `tests/mocking/` infrastructure with `MockRegistry`. Achieved >80% coverage on core modules.

### [DONE] Logging lacks structure
- **Status:** RESOLVED (Phase 5)
- **Problem:** Mixing `print` and basic `logging`.
- **Resolution:** Integrated `structlog` for JSON-formatted, contextual logging with `session_id` and `correlation_id`.

### Silent Pass Statements
- **Issue:** Multiple locations where exceptions are caught and immediately passed without any handling
- **Files:**
  - `graph_rlm/backend/src/mcp_integration/generator.py:189`: `except RuntimeError: pass`
  - `graph_rlm/backend/src/mcp_integration/core/client.py:237,362,363,480`: `except asyncio.CancelledError: pass`
  - `graph_rlm/backend/src/mcp_integration/utils/task_schema.py:176,188`: `except ImportError: pass`
- **Impact:** Failures are completely invisible to the system and operators
- **Fix approach:** Replace `pass` with logging statements and/or proper error recovery logic

### Duplicate Configuration
- **Issue:** `LLM_PROVIDER` is defined twice in `Settings` class with different default values
- **File:** `graph_rlm/backend/src/core/config.py:38,46`
- **Impact:** Unclear which value takes precedence; potential configuration bugs
- **Fix approach:** Remove duplicate definition and ensure single source of truth

### Large File Complexity
- **Issue:** Several files exceed recommended single-file complexity thresholds
- **Files:**
  - `graph_rlm/backend/src/core/agent.py` (2,292 lines)
  - `graph_rlm/backend/src/core/dream.py` (1,135+ lines)
  - `graph_rlm/backend/mcp_tools/desktop_commander.py` (1,696 lines)
- **Impact:** Difficult to maintain, test, and understand. High cognitive load for modifications
- **Fix approach:** Refactor into smaller, focused modules with single responsibilities

## Known Bugs

### No Explicit Bugs Identified
- **Note:** No explicit `FIXME` or `BUG` comments found in source code
- **Note:** The codebase uses TODO comments in desktop_commander.py for documentation purposes (search patterns)
- **Risk:** This doesn't mean bugs don't exist - the broad exception handling likely masks many failure modes

## Security Considerations

### API Key Configuration
- **Risk:** Empty default API key values in `config.py`
- **Files:** `graph_rlm/backend/src/core/config.py:49,64`
- **Current mitigation:** None - runtime failures if keys not configured
- **Recommendations:**
  - Add validation that API keys are present when LLM_PROVIDER requires them
  - Consider using secrets management instead of .env files for production

### Environment File Handling
- **Risk:** `config.py` writes directly to `.env` file including API keys
- **File:** `graph_rlm/backend/src/core/config.py:109-172`
- **Current mitigation:** Attempts to filter out some sensitive defaults
- **Recommendations:**
  - Never write API keys to .env programmatically
  - Separate configuration from secrets management

### Process Isolation
- **Positive:** Agent uses `uv` for process isolation (`graph_rlm/backend/src/core/agent.py:84`)
- **Note:** This is a security feature, not a concern

## Performance Bottlenecks

### Recursive Query Depth
- **Problem:** `MAX_RECURSION_DEPTH` is set to 3 in `config.py:43`
- **Files:** `graph_rlm/backend/src/core/config.py`
- **Cause:** Deep recursion without depth limits could cause stack overflow or infinite loops
- **Improvement path:** Consider implementing iterative approaches with explicit stack management for complex recursive operations

### Large Response Processing
- **Problem:** No apparent streaming or chunking for large LLM responses
- **Files:** `graph_rlm/backend/src/core/llm.py`
- **Cause:** Blocking wait for complete LLM responses
- **Improvement path:** Implement streaming responses and pagination for large outputs

### Database Query Performance
- **Problem:** Vector index creation happens on every GraphClient initialization
- **File:** `graph_rlm/backend/src/core/db.py:39`
- **Impact:** Slower startup time; potential duplicate index creation errors
- **Improvement path:** Check if indexes exist before creation; defer to background task

## Fragile Areas

### Skill Storage Module
- **Files:** `graph_rlm/backend/src/mcp_integration/skill_storage.py`
- **Why fragile:** Heavy use of broad exception handlers (11+ instances); dynamic skill loading; complex file operations
- **Safe modification:** Test all skill loading/unloading operations after changes
- **Test coverage:** Limited - primarily integration tests in `tests/` directory

### MCP Integration Client
- **Files:** `graph_rlm/backend/src/mcp_integration/core/client.py`
- **Why fragile:** Connection handling with multiple exception types; async cancellation; complex shutdown logic
- **Safe modification:** Test reconnection scenarios and graceful shutdown
- **Test coverage:** Basic connection tests; missing chaos testing (network failures, timeouts)

### Axiom System
- **Files:** `graph_rlm/backend/axioms_dir/`
- **Why fragile:** Many disabled axioms in `_disabled/` subdirectory (10+ disabled files); experimental verification logic
- **Safe modification:** Understand the axiom verification chain before modifying; test disabled axioms before enabling
- **Test coverage:** Axiom-specific tests scattered; no comprehensive axiom integration test suite

## Scaling Limits

### Session/Graph Memory
- **Resource:** FalkorDB graph memory
- **Current capacity:** No explicit limits configured
- **Limit:** Memory-bound by FalkorDB instance
- **Scaling path:** Implement graph cleanup policies; add TTL for old session data

### Concurrent Sessions
- **Resource:** Agent session management
- **Current capacity:** In-memory session cache in `agent.py`
- **Limit:** Memory-bound for `session_cache` dictionary
- **Scaling path:** Implement external session store (Redis) for distributed deployment

### LLM Rate Limits
- **Resource:** External LLM API calls
- **Current capacity:** No rate limiting implemented
- **Limit:** Provider-specific rate limits (OpenRouter, OpenAI, Ollama)
- **Scaling path:** Implement request queuing and rate limiting with exponential backoff

## Dependencies at Risk

### LangChain Version Pinning
- **Package:** `langchain>=1.2.7`, `langchain-community>=0.4.1`
- **Risk:** LangChain 1.x is in maintenance mode; 2.x has breaking changes
- **Impact:** Security patches and new features will eventually stop for 1.x
- **Migration plan:** Monitor LangChain 2.x migration path; plan 6-month upgrade timeline

### MCP Server Version
- **Package:** `mcp>=1.26.0`
- **Risk:** MCP is evolving rapidly; API changes likely
- **Impact:** Breaking changes could break MCP tool integrations
- **Migration plan:** Pin to specific version; test against newer versions before upgrading

### Python 3.13 Requirement
- **Package:** `requires-python = ">=3.13"`
- **Risk:** Very new Python version; some packages may lack wheels or have bugs
- **Impact:** Potential compatibility issues with packages that don't yet support 3.13
- **Migration plan:** Monitor package compatibility; consider 3.12 as fallback for stability

## Missing Critical Features

### Circuit Breaker Maturity
- **Status:** IN_PROGRESS
- **Problem:** Circuit breakers are implemented but not yet tuned for all external MCP servers.
- **Impact:** **MEDIUM**
- **Recommendation:** Monitor error rates and adjust failure thresholds per service.

### Legacy Code Cleanup
- **Status:** OPEN
- **Problem:** Legacy files like `agent.py` and `dream.py` (root level) are still present while backend implementations have moved.
- **Impact:** **LOW**
- **Recommendation:** Refactor or archive legacy files to reduce confusion.

---

*Last updated: 2026-05-04*

### Untested Critical Paths
- **What's not tested:** Agent main loop and recursive reasoning (`agent.py`)
- **File:** `graph_rlm/backend/src/core/agent.py`
- **Risk:** Logic errors in the core RLM loop go undetected
- **Priority:** Critical

### Untested Error Scenarios
- **What's not tested:** MCP server disconnections, database connection failures, LLM API timeouts
- **Files:** `graph_rlm/backend/src/mcp_integration/core/client.py`, `graph_rlm/backend/src/core/db.py`
- **Risk:** System fails silently or hangs in error conditions
- **Priority:** High

### Untested Integration Points
- **What's not tested:** Full skill execution lifecycle, axiom verification chain
- **Files:** `graph_rlm/backend/src/mcp_integration/skill_storage.py`, `graph_rlm/backend/axioms_dir/`
- **Risk:** Integration bugs between components
- **Priority:** Medium

---

*Concerns audit: 2026-02-12*