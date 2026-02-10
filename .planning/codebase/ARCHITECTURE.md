# ARCHITECTURE.md

## Overall System Pattern

**Graph-RLM** implements a **Self-Healing Recursive Language Model** with persistent graph memory. It replaces linear context windows with a topological sheaf-based approach for unbounded reasoning.

## Layer Architecture

### Layer 1: User Interface

| Component | Purpose |
|-----------|---------|
| `graph_rlm/frontend/` | React + Vite web UI |
| FastAPI REST API | Backend endpoints |

### Layer 2: Orchestration & RLM Core

| Component | Purpose |
|-----------|---------|
| Recursive Language Model (RLM) | Main orchestrator |
| Graph Memory | Persistent FalkorDB graph |
| Session Manager | REPL state management |

### Layer 3: Constraint Augmented Generation (CAG)

| Component | Purpose |
|-----------|---------|
| Document Ingestor | Parse and index documents |
| Axiom Miner | Extract logical invariants |
| Axiom Coder | Convert invariants to Python validators |
| Sheaf Monitor | Real-time axiomatic verification |

### Layer 4: Self-Healing & Sleep

| Component | Purpose |
|-----------|---------|
| Dreamer Agent | Offline consolidation |
| Reflexion Engine | Immediate error recovery |
| Ralph Protocol | Wake/Sleep cycles |

### Layer 5: External Integration

| Component | Purpose |
|-----------|---------|
| MCP Protocol | Tool/skill integration |
| LLM Adapters | OpenAI, OpenRouter, Ollama |
| Vector Store | Semantic memory |

## Data Flow

### Query Flow

```
User Input --> RLM.query() --> Graph Memory Check --> Axiom Verification --> LLM Call --> Graph Update --> Response
                                                                      |
                                                              (If violation)
                                                                      v
                                                               Reflexion
```

### Document Ingestion Flow

```
Raw Document --> Ingestor --> Miner --> Coder --> Axiom Library --> Sheaf Monitor
```

### Sleep/Dream Flow

```
High Surprise Events --> Dreamer --> Insight Extraction --> rules.md --> Next Wake Cycle
```

## Key Abstractions

### Recursive Language Model (RLM)

**Entry Point**: `rlm.query(prompt, session_id)`

**Features:**
- Recursive sub-query execution in isolated scopes
- Shared graph memory via session ID
- Unbounded depth (no context window limits)

### Graph of Thoughts (GoT)

**Representation:**
- Each thought = timestamped node
- Edges = logical connections between thoughts
- Queryable "Topological Frontier"

**Operations:**
- Add thought node
- Connect thoughts (parent/child)
- Query semantic neighborhood
- Prune low-value branches

### Axiomatic Sheaf (The Immune System)

**Components:**
1. **Axiom Library**: Python validator functions
2. **Sheaf Monitor**: Runs validators before execution
3. **Reflexion**: Self-correction on violation

**Violation Handling:**
```
Violation Detected --> SYSTEM REFLEXION Node --> Re-plan --> Re-execute
```

### Ralph Protocol (Wake/Sleep)

**Wake Phase:**
- Normal operation
- Execute tasks, accumulate experience
- Track surprise metrics

**Sleep Phase (Dreamer):**
- Query high-surprise edges
- Consolidate into insights
- Update rules.md

## Entry Points

### Backend

| File | Purpose |
|------|---------|
| `graph_rlm/backend/main.py` | FastAPI application |
| `graph_rlm/__init__.py` | RLM core interface |

### CLI Scripts

| Script | Purpose |
|--------|---------|
| `scripts/exam_cli.py` | Examination interface |
| `scripts/generate_structure.py` | Structure analysis |
| `scripts/purge_orphaned_axioms.py` | Axiom cleanup |

### Skills (MCP Tools)

| Location | Purpose |
|----------|---------|
| `skills/` | MCP tool implementations |
| `skills/*.py` | Individual skill modules |

## External Interfaces

### REST API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/query` | Recursive query |
| POST | `/ingest` | Document ingestion |
| GET | `/graph` | Graph state |
| POST | `/axiom` | Add axiom |

### Frontend Interface

| Port | Purpose |
|------|---------|
| 5173 | Vite dev server |
| 8000 | FastAPI backend |

## Key Design Patterns

### Pattern 1: Recursive Isolation

```python
rlm.query("Solve X", session_id="s1")
    └── rlm.query("Solve Y", session_id="s1")  # Inherits state
            └── rlm.query("Solve Z", session_id="s1")
```

### Pattern 2: Axiomatic Guardrails

```python
# Before any action
if not axiom.verify(proposed_action):
    trigger_reflexion("AXIOM VIOLATION")
```

### Pattern 3: Surprise-Based Learning

```python
surprise = (1 - cosine_similarity) + (axiom_violation ? 1.0 : 0.0)
if surprise > threshold:
    dreamer.consolidate_high_surprise_events()
```

## Processing Model

| Model | Characteristics |
|-------|-----------------|
| Async | Non-blocking I/O for LLM calls |
| Isolated | Separate REPL environments per session |
| Persistent | Graph state survives session boundaries |
| Recoverable | Self-healing on errors and violations |