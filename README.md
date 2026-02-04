# Self-Healing Recursive Language Model (Graph-RLM)

> **"Unshackled" Reasoning**: A system that replaces linear context windows with a persistent, recursive, and self-correcting Graph of Thoughts. Implements the **Ralph Protocol** (Wake -> Sleep -> Wake).

![Graph Visualization](image.png)

## Overview

**Graph-RLM** is an implementation of the **Recursive Language Model** paradigm. Unlike standard LLM agents that decay as context grows ($N^2$ complexity), Graph-RLM treats context as a **Topological Sheaf**.

It solves "Context Rot" through three core mechanisms:
1.  **Recursive Decomposition**: Complex tasks are broken down into sub-queries (`rlm.query()`) that execute in their own scopes but share a persistent **Graph Memory**.
2.  **Constraint Augmented Generation (CAG)**: A deterministic pivot from RAG. The system proactively **Ingests** documents, **Mines** logical invariants, and **Codifies** them into executable Python **Axioms** (Guardrails).
3.  **Sheaf Axiomatic Monitor**: A real-time verification process that runs all proposed actions against the **Axiom Library**. Violations trigger immediate Reflexion before code execution.
4.  **The Dreamer (Sleep Phase)**: An offline consolidation cycle that converts high-surprise events (failed tests, axiomatic violations) into long-term **Wisdom**.

---

## Core Architecture

### 1. The Persistent REPL & Graph Memory
Variables define state. In Graph-RLM, every session processes thoughts within a persistent **Python REPL**.
- **State Sharing**: Recursive calls (`rlm.query`) inherit the session ID, allowing sub-agents to access and modify the shared state.
- **GraphDB**: We use **FalkorDB** (Production) or **NetworkX** (Embedded) to store the **Graph of Thoughts (GoT)**. Every thought is a timestamped node, allowing us to query the "Topological Frontier" of context rather than just a linear list.

### 2. Constraint Augmented Generation (CAG)
A deterministic alternative to RAG for high-stakes domains:
- **The Miner**: Extracts absolute logical invariants (e.g., "Velocity < 300") from raw text.
- **The Coder**: Converts invariants into verified Python validator functions (Axioms).
- **The Verifier**: Automatically runs validators against proposed agent actions in a sandbox.
- **Result**: Hallucination-free execution in safety-critical environments.

### 3. Axiomatic Sheaf (The "Immune System")
- **Guardrails**: Before any code is executed, the Sheaf Monitor runs it through all relevant **Axiom Skills**.
- **Reflexion**: If an axiom is violated, the system injects a `SYSTEM REFLEXION (AXIOMATIC)` node, forcing the agent to refactor its thought.
- **Metric**: `Surprise = (1 - CosineSimilarity) + (1.0 if AxiomViolated else 0.0)`.

### 3. The Dreamer Agent (Sleep Phase)
Inspired by the **Ralph Protocol** ("Die and Repeat"):
- **Wake**: The Agent tries to solve tasks. It may fail.
- **Sleep**: The `Dreamer` module queries the graph for high-surprise edges. It uses an LLM to consolidate these failures into **Insights**.
- **Rule Injection**: These insights are appended to `rules.md`, which is injected into the System Prompt of the *next* Wake cycle. The Agent gets smarter every night.

### 4. Representation Engineering (RepE)
- **Safety Layer**: Scans thought embeddings for "Moloch" vectors (Deception, Power-Seeking) before they are written to the Graph.
- **Steering**: If a thought is unsafe, the system injects a "Reflexion" node to steer the agent back to safety.

---

## Feature Highlights

- **MCP Integration**: Fully supports the **Model Context Protocol**. Tools and "Skills" are dynamically loaded.
- **Constraint Augmented Generation (CAG)**: Built-in `ingest_document()` and `codify()` tools for automated ontology building.
- **Infinite Recursion**: Depth limits are "unshackled". The Agent can drill down indefinitely.
- **Self-Healing**: `Traceback` in the REPL or `AxiomViolation` = `High Surprise`. The system treats runtime/logic errors as semantic signals to self-correct.

---

## Tech Stack

- **Core**: Python 3.12+ (PyQt6 Desktop App)
- **Memory**: FalkorDB (Production) OR **NetworkX** (Embedded/Offline Mode)
- **Execution**: `uv` (Package Management), Native REPL
- **LLM**: OpenRouter (Cloud) or Ollama (Local/Offline)
- **Frontend**: PyQt6 (High-Performance Desktop GUI)

---

## Getting Started

### 1. Setup

```bash
# Install dependencies
pip install .
# OR
uv sync
```

### 2. Run Application

```bash
python scripts/run_app.py
```
*Note: The application automatically detects if FalkorDB is running. If not, it falls back to an embedded in-memory graph database (`graph_db.json`), allowing full offline usage without Docker.*

### 3. (Optional) Local LLM

To enable fully offline AI:
1.  Install [Ollama](https://ollama.com).
2.  Pull a model: `ollama pull tinyllama` (or your preferred model).
3.  The app will auto-detect Ollama running on localhost.

---

## License

MIT

Created by [angrysky56](https://github.com/angrysky56) with Antigravity (Gemini 2.0).
