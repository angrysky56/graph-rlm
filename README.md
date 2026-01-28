# Self-Healing Recursive Language Model (Graph-RLM)

> **"Unshackled" Reasoning**: A system that replaces linear context windows with a persistent, recursive, and self-correcting Graph of Thoughts. Implements the **Ralph Protocol** (Wake -> Sleep -> Wake).

![Graph Visualization](docs/screenshot.png)

## Overview

**Graph-RLM** is an implementation of the **Recursive Language Model** paradigm. Unlike standard LLM agents that decay as context grows ($N^2$ complexity), Graph-RLM treats context as a **Topological Sheaf**.

It solves "Context Rot" through three core mechanisms:
1.  **Recursive Decomposition**: Complex tasks are broken down into sub-queries (`rlm.query()`) that execute in their own scopes but share a persistent **Graph Memory**.
2.  **Sheaf Topology Monitor**: A background process that uses **In-Database GraphBLAS** to detect "Logical Knots" (High Surprise/Inconsistency) in real-time.
3.  **The Dreamer (Sleep Phase)**: An offline consolidation cycle that converts high-surprise events (failed tests, logical contradictions) into **Wisdom** (Rules) for the next Wake cycle.

---

## Core Architecture

### 1. The Persistent REPL & Graph Memory
Variables define state. In Graph-RLM, every session processes thoughts within a persistent **Python REPL**.
- **State Sharing**: Recursive calls (`rlm.query`) inherit the session ID, allowing sub-agents to access and modify the shared state.
- **GraphDB**: We use **FalkorDB** to store the **Graph of Thoughts (GoT)**. Every thought is a timestamped node, allowing us to query the "Topological Frontier" of context rather than just a linear list.

### 2. Sheaf Topology (The "Immune System")
We measure the **Surprise Score** (Consistency Energy) of every thought against its neighbors.
- **Metric**: `Surprise = (1 - CosineSimilarity) + (1.0 if ExecutionFailed else 0.0)`.
- **Optimization**: All calculations are pushed to FalkorDB via Cypher/GraphBLAS, handling 10k+ nodes with <10ms overhead.
- **Action**: If "Surprise" spikes, the Agent is warned of a **Logical Knot** and forced to Reflexion.

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
- **Infinite Recursion**: Depth limits are "unshackled". The Agent can drill down indefinitely.
- **Self-Healing**: `Traceback` in the REPL = `High Surprise`. The system treats runtime errors as semantic signals to change capability.

---

## Tech Stack

- **Core**: Python 3.12+ (FastAPI)
- **Memory**: FalkorDB (Graph + Vector Store)
- **Execution**: `uv` (Package Management), Native REPL
- **LLM**: OpenRouter (xAI Grok, GPT-4, etc.) or Ollama (Local)
- **Frontend**: React + Vite + D3.js (Live Graph Visualization)

---

## Getting Started

### 1. Automated Setup

```bash
./setup_env.sh    # Checks dependencies (Python, UV, Docker)
./start.sh        # Launches Database, Backend, and Frontend
```

### 2. Usage

1.  **Launch the UI** (localhost:5173).
2.  **Enter a Recursive Prompt**:
    > "Research the 'Ralph Protocol' for AI agents. Recursively break down its 5 pillars and implement a Python mock for each."
3.  **Watch the Graph**:
    - Blue Nodes: Thoughts.
    - Red Pulses: High Surprise (Logical Knots).
    - **Final Answer**: Triggers a "Micro-Dream" to save insights before exiting.

---

## License

MIT

Created by [angrysky56](https://github.com/angrysky56) with Antigravity (Gemini 2.0).
