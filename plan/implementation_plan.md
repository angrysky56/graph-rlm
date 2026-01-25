# Implementation Plan: Self-Healing Recursive Language Model (Graph-RLM)

## 1. Project Principles & Integrity Checks
*   **Reliability**: Every recursive step must be logged to the Graph DB *before* execution proceeds. State must be recoverable.
*   **Observability**: The "Graph" is not just for storage; it is the user interface. We must visualize the "thought process" in real-time.
*   **Modularity**: The REPL, the Graph, and the LLM are loose dependencies. We will use dependency injection to link them.

## 2. Architecture Specification

### 2.1 Backend (Python)
*   **Framework**: FastAPI.
*   **Database**: `falkordb` (Python client).
*   **LLM**: `ollama` and `openai` (for OpenRouter). Check docs online for the best way to do this. We aren't doing it right I don't think.
*   **Sub-Agent Interface**: Check the existing `local-repl-mcp` to execute code. Let's just port the relevant code from `local-repl-mcp` to the backend. /home/ty/Repositories/ai_workspace/local-repl-mcp

### 2.2 Frontend (React)
*   **Framework**: Vite + React + TypeScript.
*   **Visualization**: `react-force-graph` or `cytoscape.js` to render the FalkorDB nodes live.
*   **Interaction**: A chat interface that spawns the graph.

### 2.3 Data Model (Graph Schema)
*   **Node Labels**: `:Task`, `:Thought`, `:Fact`, `:Critique`.
*   **Edge Types**: `:DECOMPOSES_INTO`, `:DEPENDS_ON`, `:CONTRADICTS`, `:SUPPORTS`.
*   **Properties**: `embedding` (vector), `status` (pending/done), `energy` (consistency score).

## 3. Step-by-Step Implementation Roadmap

### Phase 1: The Graph Foundation (Infrastructure)
*   [ ] **Step 1.1**: Initialize Project Structure & `pyproject.toml`.
*   [ ] **Step 1.2**: Create `FalkorDBClient` wrapper for graph operations.
*   [ ] **Step 1.3**: Implement the `GraphLogger` that wraps LLM calls.
    *   *Goal*: When I call `llm.generate()`, a node appears in the DB.

### Phase 2: The Recursive "Primitive" (`rlm.query`) https://arxiv.org/pdf/2512.24601v1 Ensure this is aligned.
*   [ ] **Step 2.1**: Implement the `Agent` class in Python.
*   [ ] **Step 2.2**: Define `rlm.query(prompt, context_id)`:
    *   Creates a child node in FalkorDB.
    *    Calls Ollama.
    *   Updates the node with the response.
*   [ ] **Step 2.3**: Connect `local-repl-mcp` (or a direct python execution sandbox) to allow the LLM to call `rlm.query` recursively.

### Phase 3: Context Surf & Vector Integration
*   [ ] **Step 3.1**: Add embedding generation (using `ollama` embeddings or `sentence-transformers`).
*   [ ] **Step 3.2**: Store embeddings on Graph Nodes.
*   [ ] **Step 3.3**: Implement "Semantic Search" over the graph for context retrieval.

### Phase 4: The Sheaf Monitor (Diagnostics)
*   [ ] **Step 4.1**: Implement the calculation of "Consistency Energy" (Sheaf Laplacian).
    *   *Note*: This requires extracting the "latent state" (activations) if possible, or using the output embedding as a proxy for the stalk.
*   [ ] **Step 4.2**: Create the background worker that scans the graph for high-energy cycles ("Logical Knots").

### Phase 5: Immunity & Reflexion
*   [ ] **Step 5.1**: Implement the "Reflexion Trigger": If energy > Threshold, inject a critique prompt.
*   [ ] **Step 5.2**: (Advanced) RepE Hooks for PyTorch models (if local model access permits).

### Phase 6: Frontend Visualization
*   [ ] **Step 6.1**: React App scaffolding.
*   [ ] **Step 6.2**: Live polling/WebSocket to backend to stream Graph updates.
*   [ ] **Step 6.3**: Interactive Node inspection.

### Phase 7: Subsystem Agents (Function Gemma)
*   [ ] **Step 7.1**: Evaluation of FunctionGemma (via Ollama) for REPL control.
*   [ ] **Step 7.2**: Fine-tuning a small Gemma model (2B) for specific graph operations if generic models fail.
*   [ ] **Step 7.3**: "Subsystem Agent" abstraction: Specialized agents for Coding vs. Graph Management.

## 4. Test Strategy (TDD)
*   **Unit Tests**: Mock LLM responses, verify Graph topology creation.
*   **Integration Tests**: Run a real recursion (depth 2) and verify FalkorDB state.
*   **E2E**: Visual check of the graph growing during a live query.

## 5. Dependencies
*   `fastapi`
*   `uvicorn`
*   `falkordb`
*   `ollama`
*   `openai`
*   `numpy` (for Laplacian calc)
*   `networkx` (for local graph algos)

