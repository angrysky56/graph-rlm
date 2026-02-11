# Graph-RLM Source Reference

This document provides a comprehensive reference of the modules, classes, and functions within the `src` directory of the Graph-RLM backend.

---

## Package: `core`

Located at `graph_rlm/backend/src/core/`

### `cli.py` (located in `graph_rlm/backend/src/`)

**Module:** Graph-RLM Live Diagnostics Tool.
Provides interactive tests and inspections for the Agent, RepE, Sheaf, and Dreamer components.

- **Function: `test_live_repe`**: Calibrates and tests the Representation Engineering gestalt monitor.
- **Function: `test_live_sheaf`**: Verifies topological loop detection using real database nodes.
- **Function: `test_live_dreamer`**: Tests the consolidation and surprise identification cycle.
- **Function: `test_live_agent`**: Tests the full Agent execution loop.
- **Function: `inspect_session`**: Displays the scratchpad context for a given session.
- **Function: `inspect_node`**: Displays raw properties and prompt for a specific graph node.

### `agent.py`

**Module:** Recursive Logic Machine (RLM) Agent.
Handles the core execution loop, recursive querying, and tool integration.

- **Class: `Agent`**
  - The core Recursive Logic Machine (RLM) agent.
  - Handles the main execution loop, epistemic health checks, and dreamer integration.
  - Manages REPL sessions, graph memory, and tool execution.
  - **Method: `stream_query`**: Execute a query and stream events (thinking, code, output, etc.) via a queue.
  - **Method: `query_sync`**: Synchronous Recursive Logic with Stateless Graph Memory, executed in a worker thread.
  - **Method: `install_package`**: Installs a package into the active project environment.
  - **Method: `install_skill_package`**: Installs a package into the dedicated Agent/Skill virtual environment (`agent_venv`).
  - **Method: `read_skill`**: Reads the source code of a compiled skill from storage.
- **Function: `is_skills_available`**
  - Defensive check for Skills system availability.

### `config.py`

**Module:** Configuration management for Graph-RLM using Pydantic Settings.
Handles environment variables and LLM provider configurations.

- **Class: `Settings`**
  - Application configuration and environment management.
  - Handles defaults and overrides from .env and environment variables.

### `context_index.py`

**Module:** Context Indexer for Graph-RLM.
Maintains a topological summary of active contexts to prevent context rot.

- **Class: `ContextIndex`**
  - Constructs a 'Scratchpad' of active contexts (Thoughts/REPLs) to prevent context rot in the unified RLM graph.

### `core.py`

**Module:** The Python REPL implementation that maintains state between executions.
Ported from local-repl-mcp with minimal changes.

- **Class: `KnowledgeBaseStructure`**
  - Helper to provide semantic access to the KB folders.
- **Class: `PythonREPL`**
  - A stateful Python REPL implementation using isolated AgentRuntime processes.
  - Replaces the legacy 'exec' based implementation for security.

### `db.py`

**Module:** Database core module for Graph-RLM.
Handles interactions with FalkorDB, including thought node creation, vector indexing, and session management.

- **Class: `GraphClient`**
  - Client for interacting with FalkorDB.
  - Provides methods for querying, creating thoughts, and managing the graph state.

### `dream.py`

**Module:** The Dreamer: Consolidation and Wisdom layer for Graph-RLM.
Analyzes surprise events, logical knots, and codifies axioms during 'sleep' cycles.

- **Class: `Dreamer`**
  - The 'Sleep' Phase of the Graph-RLM architecture.
  - Consolidates high-entropy (Surprise) events into 'Wisdom' (Insights).
  - Also provides 'Lucid Dream' capabilities for immediate loop analysis.
  - **Method: `consolidate_session`**: Runs the full dream cycle for a session, identifying surprises and synthesizing insights.
  - **Method: `lucid_dream`**: Fast-path async critique of the current trace for immediate feedback.
  - **Method: `synthesize_axiom`**: Converts an identified logical insight into a formal Axiom node.

### `endpoints.py`

**Module:** FastAPI Endpoints for Graph-RLM.
Handles chat completions, session management, and system configuration.

- **Class: `ChatMessage`**
  - Representation of a single message in a chat conversation.
- **Class: `ChatCompletionRequest`**
  - Schema for an OpenAI-compatible chat completion request.

### `guardrails.py`

**Module:** Hardcoded guardrails and structural invariants for the Graph-RLM reasoning engine.
These validators are enforced at the database layer to prevent "Structural Orphanage" and "Null-Context Cascades".

- **Class: `GuardrailError`**
  - Exception raised when a graph invariant is violated.
- **Function: `validate_thought_node`**
  - Validates a node before it is committed to the graph.
  - Checks: Orphan Prevention, Context Continuity, Tool Causality.
- **Function: `validate_no_blind_transitions`**
  - Enforces causal semantics (e.g., TOOL_RESULT must follow TOOL_CALL).

### `llm.py`

**Module:** LLM Service Layer for Graph-RLM.
Provides a unified interface for multiple LLM providers and embedding models.

- **Class: `LLMService`**
  - Unified LLM client supporting OpenRouter, Ollama, and OpenAI-compatible endpoints.
  - Uses a persistent httpx.AsyncClient for connection pooling and robust timeout management.

### `log_stream.py`

**Module:** Log streaming infrastructure for real-time terminal output to frontend.
Captures all backend logs and streams them via WebSocket.

- **Class: `LogBuffer`**
  - Thread-safe buffer that captures log messages for streaming.
  - Maintains a fixed-size history and broadcasts to connected clients.
- **Class: `StreamingHandler`**
  - Custom logging handler that feeds logs to the LogBuffer.
- **Function: `setup_log_streaming`**
  - Configure the root logger to stream all logs to the buffer.

### `logger.py`

**Module:** Structured Logging utility with ANSI color support for Graph-RLM.

- **Function: `get_logger`**
  - Returns a structured logger.

### `manager.py`

**Module:** REPL Manager for Graph-RLM.
Manages lifecycle and state of multiple PythonREPL instances.

- **Class: `REPLManager`**
  - Manages multiple PythonREPL instances.

### `mcp_runtime.py`

**Module:** MCP Server discovery and proxy logic for the Graph-RLM Agent.

- **Class: `MCPServerNamespace`**
  - Lazy-loaded namespace for a single MCP server.
- **Class: `LazyMCPNamespace`**
  - Lazy-loaded root namespace for all MCP servers.
- **Function: `is_mcp_available`**
  - Defensive check for MCP tools availability.
- **Function: `is_skills_available`**
  - Defensive check for skills/manager availability.

### `meta_agents.py`

**Module:** Meta-Agent Collaboration Framework.
Implements the Breaker/Synthesizer pattern for recursive Sub-REPL collaboration.

- **Class: `AgentRole`**
  - Role of a Sub-REPL in the collaboration.
- **Class: `Fragment`**
  - Result from a Breaker Sub-REPL.
- **Class: `CollaborationState`**
  - State of a Breaker/Synthesizer collaboration.
- **Class: `MetaAgentController`**
  - Orchestrates Breaker/Synthesizer collaboration for complex tasks.

### `monitor.py`

**Module:** Background Monitor for Graph-RLM.
Periodically scans the thought graph for drift and consistency issues.

- **Class: `BackgroundMonitor`**
  - Orchestrates the periodic execution of system monitoring tasks.
  - Runs the SheafMonitor in a background thread to analyze system energy profiles and thought graph consistency.

### `navigator.py`

**Module:** Navigator: The Engine of Curiousity for Graph-RLM.
Implements Intrinsic Motivation via Compression Progress and Causal Entropic Forces.

- **Class: `Navigator`**
  - The Navigator is the agent's active explorer.
  - It ranks potential actions based on their 'Interestingness' (Curiosity Score).
  - Metrics: Compression Progress, Causal Entropic Force, Topological Consistency.
  - **Method: `rank_actions`**: Rank a set of potential actions based on curiosity metrics.
  - **Method: `evaluate_curiosity`**: Calculate a combined curiosity score for a state or action.

### `navigator_config.py`

**Module:** Configuration for the Sheaf-Theoretic Navigator.
Hyperparameters for curiosity-driven exploration and topological analysis.

### `omcd.py`

**Module:** oMCD (online Metacognitive Control of Decisions) Controller.
Implements optimal resource allocation and stopping decisions for the Agent.

- **Class: `OmcdParams`**
  - Tunable parameters for the oMCD model.
- **Class: `OmcdController`**
  - Metacognitive Controller for Optimal Decision Making.

### `prompts.py`

**Module:** System prompt templates for the Graph-RLM Agent.

### `reflexion.py`

**Module:** Reflexion/IntelliSynth Module.
Implements the IntelliSynth Framework for breaking logical knots and stagnation.

- **Class: `IntelliSynth`**
  - Implements Analyze with Logic (AwL), Advancement Cycle, and Mathematical Context.

### `repe.py`

**Module:** Representation Engineering (RepE) v2: Gestalt Vector Monitor.
Provides psychological profiling of agent thoughts using steering axes.

- **Class: `GestaltMonitor`**
  - RepE v2: Gestalt Vector Monitor.
  - Calculates 'Steering Axes' based on Fritz Perls' continuum of neurosis.
  - **Method: `analyze_thought`**: Analyzes a thought string for various psychology-inspired steering axes (Laziness, Malice, etc.).
  - **Method: `get_gestalt_score`**: Aggregate multiple axis scores into a single system health metric.

### `rlm_interface.py`

**Module:** The RLM interface exposed to the agent REPL as 'rlm'.

- **Class: `RLMInterface`**
  - The object exposed to the REPL as 'rlm'.
  - Allows recursive queries and memory recall.

### `scratchpad_builder.py`

**Module:** Scratchpad Builder for Stateless Agent Context.
Constructs a compact, actionable scratchpad for the agent.

- **Class: `ScratchpadBuilder`**
  - Builds a structured scratchpad for the stateless agent.

### `sheaf.py`

**Module:** Sheaf Monitor: Topological Field Analyzer and Axiomatic Consistency Checker.
Provides diagnostics for holonomy (loop detection) and teleology (drift detection).

- **Class: `SheafMonitor`**
  - SheafMonitor v2 (Self-Healing): Topological Field Analyzer.
  - Monitors the 'Contact Boundary' between the Agent's trajectory and the Goal.
  - **Method: `measure_consistency_energy`**: Calculates the 'energy' of a logic chain based on branch overlap and loop density.
  - **Method: `detect_holonomy`**: Identifies loops or redundant logic paths in the graph.
  - **Method: `validate_axioms`**: Checks recent thoughts against the global Axiom library.

### `state.py`

**Module:** Shared execution state and tracing utilities for the Graph-RLM Agent.

- **Class: `ExecutionState`**
  - Thread-local state for the agent's execution loop.
- **Function: `broadcast_trace`**
  - Monitor callback to push trace logs to the active event loop.

### `trace.py`

**Module:** Observability and Tracing layer for Graph-RLM.
Provides high-fidelity logging of agent actions and system state transitions.

- **Function: `register_monitor`**
  - Registers a callback to receive trace logs in real-time.
- **Function: `trace_action`**
  - High-fidelity tracing for system-wide observability.
- **Function: `banner`**
  - Prints a bright banner to find transitions easily in logs.

---

## Package: `mcp_integration`

Located at `graph_rlm/backend/src/mcp_integration/`

### `client.py`

**Module:** Coordinator client for calling actual MCP servers.
Provides the runtime communication layer between generated tool wrappers and real MCP servers.

- **Class: `CoordinatorClient`**
  - Client for communicating with MCP servers at runtime.

### `config.py`

**Module:** Configuration management for MCP-Coordinator.
Handles loading MCP server configuration from multiple sources.

- **Class: `ConfigManager`**
  - Manages MCP server configuration with multiple loading strategies.
- **Function: `create_default_env_file`**
  - Create a default .env file.

### `discovery.py`

**Module:** MCP server discovery and introspection.
Connects to MCP servers and discovers their capabilities (tools, schemas, etc.).

- **Class: `MCPServerConfig`**
  - Configuration for a single MCP server.
- **Class: `ServerIntrospector`**
  - Introspects MCP servers to discover capabilities.
- **Class: `ConfigLoader`**
  - Loads MCP server configurations.
- **Function: `discover_tools`**
  - Synchronous wrapper to get tool names for all servers.
- **Function: `find_tools`**
  - Find tools matching a semantic query.

### `generator.py`

**Module:** Python code generator for MCP tool wrappers.
Creates importable Python modules from discovered MCP server capabilities.

- **Class: `ToolGenerator`**
  - Generates Python wrapper code for MCP tools.
- **Function: `generate_from_config`**
  - High-level function to generate tools from config.

### `kernel.py`

**Module:** Persistent Python kernel for isolated execution.
Handles JSON commands over stdin/stdout and maintains state.

- **Class: `IPCClient`**
  - Mock MCP client that proxies tool calls over IPC.
- **Class: `RLMClient`**
  - Mock RLM client for proxying agent interface calls.

### `runtime.py`

**Module:** Runtime environment manager for the isolated agent.
Handles virtual environment resolution, subprocess execution, and IPC.

- **Class: `AgentRuntime`**
  - Manages the 'agent_venv' execution context using 'uv run'.
- **Function: `get_stop_event` / `set_stop_event`**
  - Global stop event management.

### `skill_harness.py`

**Module:** Skill execution harness.
Allows executing skills from the database via CLI or as isolated subprocesses.

- **Class: `MCPProxy` / `RLMProxy`**
  - Proxies for 'mcp' and 'rlm' namespaces inside skills to ensure they can call tools and recurse.
- **Function: `ensure_skills_venv`**
  - Synchronous utility to ensure the `agent_venv` exists and has core dependencies.
- **Function: `run_skill_isolated`**
  - Runs a skill in a dedicated process using the `AgentRuntime`.

### `skill_storage.py`

**Module:** Skills persistence system.
Enables AI to save and reuse learned patterns using FalkorDB.

- **Class: `SkillsManager`**
  - Manages a directory of reusable skills persisted in FalkorDB and synced to disk.
  - **Method: `sync_from_disk`**: Scans the `skills/` directory and upserts new/modified skills into the graph.
  - **Method: `save_skill`**: Persists a new Python skill to both disk and FalkorDB.
  - **Method: `get_skill`**: Retrieves skill code and metadata from the graph.
- **Class: `AxiomsManager`**
  - Manages a directory of auto-generated axioms (validators) in FalkorDB.
  - **Method: `add_axiom`**: Adds a new formal rule to the axiom library.
  - **Method: `find_similar_axioms`**: Performs vector search to find axioms relevant to a specific thought.

### `utils.py`

**Module:** Utilities for MCP integration.

- **Function: `normalize_mcp_result`**
  - Normalizes MCP tool results to make them more agent-friendly.

### Sub-package: `mcp_integration/core`

- **`client.py`**: MCP Client Manager with state machine architecture for lazy loading.
  - **Class: `McpClientManager`**: Lazy-loading MCP client manager.
- **`config.py`**: Configuration models using Pydantic for MCP Coordinator.
  - **Class: `McpConfig`**: Root configuration for all MCP servers.

### Sub-package: `mcp_integration/utils`

- **`schema_builder.py`**: Implementation of the 'Schema of Schema' concept.
  - **Class: `SchemaBuilder`**: Create, instantiate, infer, and evolve schemas.
  - **Method: `create_schema`**: Define a new schema with support for inheritance.
  - **Method: `evaluate_fit`**: Calculate how well data fits a schema (Assimilation vs. Accommodation).
  - **Method: `bootstrap_meta_ontology`**: Initializes the core schemas (Class, Property, etc.) used to bootstrap the system.
- **`task_schema.py`**: Schema-Guided Task Processing Utility.
  - **Class: `TaskSchemaProcessor`**: Coordinator for schema-guided task processing.
  - **Method: `classify_task`**: Categorizes a natural language task into one of the known schemas (Search, Code, Reasoning).
  - **Method: `suggest_tools`**: Returns a prioritized list of tools based on the identified task category.
- **Function: `classify_and_route_task`**
  - Top-level convenience function to classify a task and get tool recommendations in one call.
