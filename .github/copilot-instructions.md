# Copilot Instructions for Graph-RLM

**Graph-RLM** is a Self-Healing Recursive Language Model with persistent graph memory, combining three core paradigms: Recursive Decomposition, Constraint Augmented Generation (CAG), and Sheaf Axiomatic Monitoring.

## Architecture Overview

### Core Components

1. **Agent** ([agent.py](graph_rlm/backend/src/core/agent.py))
   - Main reasoning loop with `query_sync()` entry point
   - Exposes `RLMInterface` (rlm) in REPL for recursive queries, memory recall, axiom ingestion
   - Streaming via `stream_query()` using queue-based event emission
   - Session-based state isolation with depth tracking (prevents infinite recursion)

2. **Graph Memory** ([db.py](graph_rlm/backend/src/core/db.py))
   - FalkorDB stores thought nodes (UUID, content, embedding, parent_id, session_id, status)
   - Cypher queries access topological frontier for context synthesis
   - Vector indexes enable semantic search via `rlm.recall(query)`

3. **The Sheaf Monitor** ([sheaf.py](graph_rlm/backend/src/core/sheaf.py))
   - Pre-execution axiomatic consistency check: `check_axiomatic_consistency(code)`
   - Detects axiom violations before code runs (prevents runtime errors)
   - Returns diagnostic with status (PASS/AXIOMATIC_VIOLATION/ERROR)

4. **RepE Safety Layer** ([repe.py](graph_rlm/backend/src/core/repe.py))
   - Scans thought embeddings for "pathogenic" patterns (Laziness, Obsequiousness, Reward-Hacking, Malice)
   - Triggers reflexion if cosine similarity to antigens exceeds threshold (0.85)
   - Content filtering blocks flagged keywords before thought storage

5. **Dreamer (Sleep Cycle)** ([dream.py](graph_rlm/backend/src/core/dream.py))
   - Post-query consolidation: queries graph for high-surprise edges
   - Extracts insights from failures and appends to `rules.md`
   - Injected back into system prompt next session (continuous learning)

### Data Flow

```
Prompt → query_sync() → [Generate Code] → [Sheaf Check] → [RepE Scan]
  → [REPL Execute] → [Emit Events] → stream_query() → UI
  → [Post-Query Dream] → rules.md Update
```

## Key Patterns & Conventions

### Recursive Queries & Session Isolation
- `rlm.query(prompt, context?)` spawns **child agent** with incremented depth, new session_id
- Child inherits same REPL namespace but operates in isolated thought graph
- **Important**: Use `session_id` to link thoughts; `root_session_id` preserves request ancestry
- Example: [agent.py#L237](graph_rlm/backend/src/core/agent.py#L237) RLMInterface.query()

### Axiom Management (CAG Paradigm)
- Axioms are validator functions saved to `skills_dir/` via `rlm.save_skill(name, code, desc, tags=['axiom'])`
- Before execution, sheaf discovers relevant axioms using `_detect_required_axioms_agentic(prompt, code)`
- Violating code triggers reflexion: agent rewrites approach before retry
- Example: [test_cag_basics.py](tests/test_cag_basics.py#L1) demonstrates axiom blocking

### Thought Node Structure
Every thought creates a DB node with:
- **UUID**: Unique identifier
- **Content**: Raw thought text
- **Embedding**: Vector for semantic search (generated post-LLM)
- **Status**: "success", "error", "axiom_violation", "stall_detected"
- **Parent/Session/Root IDs**: For graph traversal

### Testing Patterns
- Tests use `asyncio.run()` wrappers for async code
- Mock DB/LLM for unit tests; use real DB for integration tests (requires FalkorDB running)
- Example patterns:
  - [test_agent_manual.py](graph_rlm/backend/tests/test_agent_manual.py): Mock agent, real REPL
  - [test_cag_basics.py](tests/test_cag_basics.py): Live axiom validation
  - [verify_repl_v2.py](tests/verify_repl_v2.py): REPL async execution

## Critical Workflows

### Setup & Launch
```bash
./setup_env.sh      # Create venvs, install deps
./start.sh          # Start FalkorDB, Backend API (port 8000), Frontend (port 5173)
```

### Backend Development
- **Async Context**: Everything is async; use `await` liberally
- **Thread Safety**: Agent runs `query_sync()` in a worker thread; use queue for event emission
- **Configuration**: `graph_rlm/backend/src/core/config.py` loads from `.env` or OS vars
- **MCP Integration**: Tools auto-discovered from `mcp_servers.json` on startup
  - Generated tool wrappers in `mcp_tools/` (auto-regenerated if config changes)
  - Access via `mcp.<server>.<tool>` in REPL (lazy-loaded namespace)

### Common Debugging Scenarios

**Agent hangs or doesn't respond:**
- Check `agent_debug.log` for LLM response (logs raw system prompt and response)
- Verify REPL is alive: `self.repl_manager.get_repl(repl_id)` returns non-None
- Inspect DB: `db.query("MATCH (t:Thought {session_id: '...'}) RETURN t LIMIT 5")`

**Code executes but axiom blocks it:**
- Sheaf diagnostic prints violation reason and suggested axiom name
- Check `skills_dir/` for axiom definition
- Use `rlm.help()` in REPL to list available axioms

**Vector search returns nothing:**
- Embeddings fail silently if LLM is down; check `llm.get_embedding(text)` directly
- Verify FalkorDB indexes: [db.py#L50](graph_rlm/backend/src/core/db.py#L50) `create_vector_indexes()`

## Code Organization

```
graph_rlm/backend/
├── src/core/
│   ├── agent.py           # Agent loop, RLMInterface, recursive primitives
│   ├── db.py              # GraphClient (FalkorDB CRUD)
│   ├── sheaf.py           # Axiomatic consistency checking
│   ├── repe.py            # Safety (embeddings + antigens)
│   ├── dream.py           # Insight consolidation
│   ├── llm.py             # LLM service (Ollama/OpenRouter)
│   ├── core.py            # PythonREPL (sandboxed exec)
│   ├── endpoints.py       # FastAPI routes (/query, /stream, etc.)
│   └── config.py          # Settings (env-based)
├── src/mcp_integration/   # MCP discovery, skills loading
├── tests/                 # Unit tests (mock-based)
└── skills_dir/            # Persisted axioms & custom skills
```

## Common Pitfalls & Fixes

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `ImportError: mcp_tools` | MCP discovery not run | Restart backend or manually trigger discovery |
| Axiom blocks valid code | Overly strict validator | Update axiom in `skills_dir/` |
| REPL namespace loses variable | Session mismatch | Pass correct `session_id` to recursive calls |
| Embedding lookup returns empty | DB index stale | Run `db.create_vector_indexes()` at startup |
| Dreamer timeout (60s) | Graph too large | Limit surprise query to recent edges |

## Conventions This Project Uses

- **No Optional Parameters**: Prefer explicit args; use defaults conservatively
- **Session IDs**: Always propagate in recursive calls; use UUID format
- **Error Handling**: Log, emit event, create error node—don't silently fail
- **Async/Await**: If function returns awaitable, must be awaited or passed to caller
- **Type Hints**: All functions have `-> Type` hints for clarity
