# Architecture

**Analysis Date:** 2026-02-12

## Pattern Overview

**Overall:** Graph-Based Recursive Language Model (RLM) with Sheaf-Theoretic Monitoring

**Key Characteristics:**
- **Graph Memory Architecture**: Context is stored as a persistent FalkorDB graph rather than linear context windows. Thoughts are nodes connected by `DECOMPOSES_INTO` edges.
- **Recursive Query Engine**: Tasks decompose into sub-queries (`rlm.query()`) that execute in isolated scopes while maintaining shared graph state.
- **Constraint Augmented Generation (CAG)**: Documents are ingested, mined for invariants, and codified into executable Python axioms (guardrails) that validate behavior.
- **Three-Tier Self-Healing System**: Tier 1 (Innate Immunity - dependency healing), Tier 2 (Epistemic Integrity - RepE safety monitoring), Tier 3 (Adaptive Immunity - Dreamer offline consolidation).
- **Ralph Protocol (Wake/Sleep/Wake)**: Agents alternate between active reasoning (Wake) and offline consolidation (Sleep/Dreamer).

## Layers

**API Layer:**
- Purpose: HTTP/WebSocket interface for frontend communication
- Location: `graph_rlm/backend/main.py`, `graph_rlm/backend/src/core/endpoints.py`
- Contains: FastAPI app, REST endpoints for chat/sessions/system, WebSocket for log streaming
- Depends on: Agent layer, LLM service, Graph database
- Used by: Frontend React/Vite application

**Agent Layer (Recursive Logic Machine):**
- Purpose: Core reasoning engine with recursive query execution and tool orchestration
- Location: `graph_rlm/backend/src/core/agent.py`
- Contains: `Agent` class, execution loop, stream_query method, REPL management, session state
- Depends on: Graph database, LLM service, Skills/Axioms managers, Sheaf monitor, Navigator
- Used by: API layer, CLI

**LLM Integration Layer:**
- Purpose: Unified interface for multiple LLM providers (OpenRouter, Ollama, LM Studio, OpenAI)
- Location: `graph_rlm/backend/src/core/llm.py`
- Contains: Provider configuration, embedding generation, model routing
- Depends on: External LLM APIs, Configuration settings
- Used by: Agent layer, Scratchpad builder, Embedding service

**Graph Database Layer (FalkorDB):**
- Purpose: Persistent storage for thoughts, skills, axioms, and round metadata
- Location: `graph_rlm/backend/src/core/db.py`
- Contains: `GraphClient` class, Cypher query methods, vector indexing, thought node management
- Depends on: FalkorDB server, Configuration settings
- Used by: All layers requiring state persistence

**MCP Integration Layer:**
- Purpose: Model Context Protocol server discovery, tool generation, and execution
- Location: `graph_rlm/backend/src/mcp_integration/`
- Contains: `skill_storage.py` (skills/axioms sync), `runtime.py` (process isolation), `generator.py` (tool generation), `discovery.py` (server discovery)
- Contains: `graph_rlm/backend/mcp_tools/` (generated wrapper modules for MCP servers)
- Depends on: MCP servers (configured in `mcp_servers.json`), Agent runtime, Graph database
- Used by: Agent layer (tool execution)

**Skills & Axioms Layer:**
- Purpose: Reusable skill library and constraint validators (Constraint Augmented Generation)
- Location: `graph_rlm/backend/axioms_dir/`, `skills/`
- Contains: Python functions as skills, Python validator functions as axioms
- Depends on: Graph database (for storage), LLM service (for embedding)
- Used by: Agent layer (skill invocation), Guardrail validation

**Monitoring & Safety Layer:**
- Purpose: Real-time verification, epistemic integrity, and topological analysis
- Location: `graph_rlm/backend/src/core/sheaf.py`, `graph_rlm/backend/src/core/repe.py`, `graph_rlm/backend/src/core/dream.py`
- Contains: Sheaf consistency monitoring, RepE safety (Gestalt) monitoring, Dreamer offline consolidation
- Depends on: Graph database (thought metrics), Agent state
- Used by: Agent layer (pre-execution validation)

**Context Management Layer:**
- Purpose: Scratchpad compression and context frontier retrieval
- Location: `graph_rlm/backend/src/core/scratchpad_builder.py`, `graph_rlm/backend/src/core/context_index.py`
- Contains: Dynamic context manager, thought retrieval, scratchpad generation
- Depends on: Graph database, LLM service (for summarization)
- Used by: Agent layer (context loading)

## Data Flow

**User Query Flow:**

1. **Request Entry**: User sends prompt via `/api/v1/chat/completions` endpoint
2. **Session Management**: Agent generates or reuses session_id (root_session_id)
3. **Context Loading**: Scratchpad builder retrieves relevant thoughts from graph frontier
4. **Prompt Construction**: System prompt + scratchpad + user query → LLM
5. **Reasoning Loop**:
   - LLM generates thought/potential action
   - Guardrail validation via axioms (if applicable)
   - Sheaf consistency check
   - Tool execution (if needed) via MCP runtime
   - Result storage in graph as Thought node
6. **Termination**: Agent returns final response when task complete
7. **Sleep Phase**: Dreamer analyzes failed trajectories for axiom synthesis

**Skill Invocation Flow:**

1. **Discovery**: Agent queries FalkorDB for relevant skills by embedding similarity
2. **Retrieval**: Skills manager loads skill metadata from graph
3. **Execution**: Skill function imported and called within isolated REPL (`agent_venv`)
4. **Storage**: Execution result logged to graph

**Axiom Validation Flow:**

1. **Trigger**: Before critical operations (file I/O, thought creation), guardrails check
2. **Validation**: Axiom function executed with context parameters
3. **Result**: Valid → proceed; Invalid → GuardrailError raised
4. **Synthesis**: Failed validations fed to Dreamer for new axiom creation

**State Management:**
- **Session State**: Stored in `ExecutionState` (thread-local) plus persistent graph
- **Graph State**: Thoughts, Rounds, Skills, Axioms all persisted in FalkorDB
- **Context Compression**: Scratchpad builder compresses graph nodes into LLM context
- **Round Archiving**: Completed sessions saved as `Round` nodes for compressed reference

## Key Abstractions

**Thought Node:**
- Purpose: Represents a single reasoning step in the graph
- Location: Created via `GraphClient.create_thought_node()` in `graph_rlm/backend/src/core/db.py`
- Properties: id, prompt, result, status, session_id, root_session_id, sheaf_score, spectral_energy, h0_rank, etc.
- Pattern: Node in FalkorDB with vector embedding on prompt

**Skill:**
- Purpose: Reusable Python function persisted in graph and disk
- Location: `skills/` directory, `Skill` nodes in graph
- Pattern: AST-parsed Python file, stored as metadata with embedding

**Axiom:**
- Purpose: Validator function enforcing constraints (CAG)
- Location: `graph_rlm/backend/axioms_dir/` directory, `Axiom` nodes in graph
- Pattern: Python function returning bool or raising GuardrailError

**REPL Session:**
- Purpose: Isolated Python execution environment for code generation
- Location: Managed by `AgentRuntime` in `graph_rlm/backend/src/mcp_integration/runtime.py`
- Pattern: `uv` subprocess with isolated virtual environment (`agent_venv`)

**Graph Client:**
- Purpose: Unified interface to FalkorDB
- Location: `graph_rlm/backend/src/core/db.py`
- Pattern: Wrapper around FalkorDB client with Graph of Thoughts schema

## Entry Points

**Backend Server:**
- Location: `graph_rlm/backend/main.py`
- Triggers: `uvicorn graph_rlm.backend.main:app`
- Responsibilities: FastAPI app initialization, MCP tool generation, RepE calibration, Skills/Axioms sync, Log streaming setup

**Chat Completions:**
- Location: `POST /api/v1/chat/completions` in `graph_rlm/backend/src/core/endpoints.py`
- Triggers: Frontend user prompt
- Responsibilities: Agent.query() execution, SSE response streaming

**CLI Interface:**
- Location: `graph_rlm/backend/src/cli.py`
- Triggers: `python -m graph_rlm.backend.src.cli`
- Responsibilities: Command-line interaction, script execution

## Error Handling

**Strategy:** Multi-layered error handling with recovery and synthesis

**Patterns:**
- **Guardrail Errors**: `GuardrailError` raised by axiom validation (Tier 1)
- **RepE Violations**: Gestalt monitoring triggers halt on psychological pathogens (Tier 2)
- **Sheaf Violations**: Topological consistency failure triggers recalibration
- **REPL Errors**: Syntax/execution errors captured and fed to Dreamer (Tier 3)
- **Session Isolation**: Errors in sub-queries don't crash main agent

## Cross-Cutting Concerns

**Logging:** Structured logging via `get_logger()` in `graph_rlm/backend/src/core/logger.py`; WebSocket log streaming to frontend

**Validation:** Guardrail system in `graph_rlm/backend/src/core/guardrails.py`; Axiom validators in `axioms_dir/`

**Authentication:** Not implemented (local development focus); API keys via environment variables

**Configuration:** Pydantic Settings in `graph_rlm/backend/src/core/config.py`; .env file support

**Process Isolation:** `uv` managed virtual environments via `AgentRuntime` in `mcp_integration/runtime.py`

---

*Architecture analysis: 2026-02-12*
