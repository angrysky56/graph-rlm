# Self-Healing Recursive Language Model (Graph-RLM)

> **"Unshackled" Reasoning**: A system that replaces linear context windows with a persistent, recursive, and self-correcting Graph of Thoughts.

## Overview

**Graph-RLM** is an implementation of the **Recursive Language Model** paradigm. Unlike standard LLM agents that try (and fail) to cram entire documents into a single prompt, Graph-RLM treats context as an **Execution Graph**.

It solves "Context Rot" and "Truncation" through two core mechanisms:

1.  **Recursive Decomposition**: Complex tasks are broken down into sub-queries (`rlm.query()`) that execute in their own scopes but share a persistent memory.
2.  **The Self-Healing Loop**: A biological-inspired "Reflexion" cycle. If the Agent encounters an error (syntax, missing tool, logic flaw), it does not crash. It **Reflects** on the error trace, updates its plan, and **Retries**.

---

## Core Architecture

### 1. The Persistent REPL (The "Hippocampus")

Variables define state. In Graph-RLM, every session processes thoughts within a persistent **Python REPL**.

- **Significance**: When the Agent reads a paper in Step 1 (`paper = read(...)`), that variable `paper` is available in Step 10, even if the LLM's context window has flushed.
- **State Sharing**: Recursive calls (`rlm.query`) inherit the session ID, allowing sub-agents to access and modify the shared state.

### 2. The Self-Healing Loop (The "Immune System")

We do not trust the LLM to write perfect code. We trust the **Loop** to fix it.

```python
while step < max_steps:
    1. Think: "I need to search Arxiv."
    2. Act: Execute `arxiv.search(...)`
    3. Observe:
       - SUCCESS: Continue.
       - FAILURE (e.g., AttributeError or Truncation):
         "Error: 'arxiv' has no attribute 'search'."
    4. Reflect: "I used the wrong method name. I should use 'search_papers'."
    5. Retry: Re-execute with corrected code.
```

_This allows the system to recover from hallucinations and Model Collapse artifacts automatically._

### 3. The Temporal Graph (FalkorDB)

Every thought is a node. Every dependency is an edge.

- **Graph Database**: We use **FalkorDB** to store the **Graph of Thoughts (GoT)**.
- **Time-Travel**: Since every thought is timestamped and linked (`DECOMPOSES_INTO`), we can traverse the history of "Why did I think this?"—turning "Chat History" into a **Causal Graph**.

---

## Feature Highlights

- **Runtime Tool Injection**: MCP Tools are injected dynamically. If the generic `arxiv` module is missing, the Agent auto-aliases `arxiv_mcp_server` to `arxiv` at runtime.
- **Infinite Recursion**: Depth limits are "unshackled" (soft limit 100). The Agent can drill down indefinitely (Abstract -> Section -> Paragraph).
- **Truncation Recovery**: If the LLM output is cut off (middle of a code block), the Agent extracts the partial block, catches the inevitable `SyntaxError`, and retries with a directive to "Complete the code".

---

## Tech Stack

- **Core**: Python 3.10+ (FastAPI)
- **Memory**: FalkorDB (Graph + Vector Store)
- **Execution**: Built-in `code` module (Sandboxed REPL)
- **LLM**: OpenRouter (xAI Grok, GPT-4, etc.) or Ollama (Local)
- **Frontend**: React + Vite (Graph Visualization)

---

## Getting Started

### 1. Automated Setup

```bash
./setup_env.sh    # Checks dependencies (Python, UV, Docker)
./start.sh        # Launches Database and API
```

### 2. Manual Setup

**Requirements**: Python 3.10+, Docker (for FalkorDB), `uv` package manager.

1.  **Start Database**:
    ```bash
    docker run -p 6379:6379 -it --rm falkordb/falkordb
    ```
2.  **Install Backend**:
    ```bash
    cd backend
    uv pip install -r requirements.txt
    ```
3.  **Run API**:
    ```bash
    uvicorn main:app --reload
    ```

---

## Usage Example: Recursive Summarization

To test the system's ability to handle large contexts:

1.  **Launch the UI** (localhost:5173).
2.  **Enter a Recursive Prompt**:
    > "I have a large file at `/path/to/paper.pdf`. Read the introduction, then recursively decide which section to read next to answer: 'Does this architecture beat Transformers?'"
3.  **Watch the Graph**:
    - You will see the Root Thought spawn Child Thoughts.
    - If the Agent makes a mistake (e.g., bad import), you will see a **Self-Healing** event in the logs as it corrects itself.

---

Resources:

- [Recursive Language Models](https://arxiv.org/pdf/2512.24601v1)
- [Designing a Self-Healing RLM Framework](docs/Designing-a-Self-Healing-RLM-Framework.md)
- [FalkorDB](https://github.com/falkordb/falkordb)
-

## License

MIT

Created by [ngrysky56](https://github.com/angrysky56) Gemini Pro 3.0 and Claude Opus 3.5
