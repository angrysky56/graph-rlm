# STRUCTURE.md

## Directory Layout

```
graph-rlm/
├── .planning/                   # GSD planning artifacts
│   └── codebase/                # This codebase map
├── .trunk/                      # Linter configuration
│   └── configs/
│       ├── .markdownlint.yaml
│       ├── ruff.toml
│       └── .yamllint.yaml
├── .vscode/                     # VS Code settings
│   └── settings.json
├── .github/                     # GitHub configuration
│   └── FUNDING.yml
├── graph_rlm/                   # Core package
│   ├── __init__.py              # Main RLM exports
│   ├── backend/                 # FastAPI backend
│   │   ├── __init__.py
│   │   └── main.py
│   └── frontend/                # React frontend
│       ├── package.json
│       ├── package-lock.json
│       ├── tsconfig.json
│       ├── tsconfig.app.json
│       ├── tsconfig.node.json
│       └── (React app source)
├── scripts/                    # Utility scripts
│   ├── exam_cli.py
│   ├── generate_structure.py
│   ├── purge_orphaned_axioms.py
│   ├── regenerate_tools.py
│   └── verify_curiosity.py
├── skills/                      # MCP skill implementations
│   ├── __init__.py
│   ├── ask_gordon.py
│   ├── atmospheric_entropy.py
│   ├── auto_crystallize_insights_skill.py
│   ├── auto_decay_ruminator_skill.py
│   ├── bootstrap_environment.py
│   ├── calculate_marathon_pace.py
│   ├── categorical_kg_bridge.py
│   ├── check_return_type.py
│   ├── cognitive_research.py
│   ├── coherence_shield.py
│   ├── commit_epistemic_victory_skill.py
│   ├── coordinator_enhanced_task.py
│   ├── crystallize_epistemic_victory.py
│   ├── daily_knowledge_digest.py
│   ├── debug_path.py
│   ├── debug_team.py
│   ├── demo_math.py
│   ├── difficult_problem_solver.py
│   ├── dikw_modeler.py
│   ├── docker_ps_raw.py
│   ├── docker_safe_write.py
│   ├── entropy_walker_experiment.py
│   ├── epistemic_agency_loop_skill.py
│   ├── epistemic_renormalization_protocol_v3.py
│   ├── file_writer.py
│   ├── firecrawl_web_search.py
│   ├── generate_adapters.py
│   ├── get_wolfram_simple_answer.py
│   ├── higuchi_fractal_dimension.py
│   ├── hybrid_cognitive_search_skill.py
│   ├── list_mcp_containers.py
│   ├── logic_gated_ingestion_final.py
│   ├── logic_gated_ingestion_v2.py
│   ├── market_analysis.py
│   ├── ralph_protocol.py
│   ├── ralph_v1.py
│   ├── replay_entropy_trace_skill.py
│   ├── research.py
│   ├── run_code_agency.py
│   ├── run_physics_screener.py
│   ├── safe_write_file.py
│   ├── save_entropy_trace_skill.py
│   ├── schema_builder.py
│   ├── schema_category_integration.py
│   ├── scientific_orchestrator.py
│   ├── search_web.py
│   ├── snapshot_crypto.py
│   ├── test_requests.py
│   ├── test_skill_1.py
│   ├── test_vg_skill.py
│   ├── track_alpha_signals.py
│   ├── validate_system_health.py
│   ├── verify_barber_paradox.py
│   ├── verify_coinbase_tradability.py
│   ├── verify_epistemic_integrity.py
│   ├── wiki_extract.py
│   ├── wolfram_query.py
│   └── xuanji_proof.py
├── tests/                       # Test files
│   ├── __init__.py
│   ├── check_dim.py
│   ├── debug_llm_response.py
│   ├── debug_paper_read.py
│   ├── debug_prompt.py
│   ├── diagnose_overlap.py
│   ├── get_axiom_code.py
│   ├── inspect_db.py
│   ├── list_skills_db.py
│   ├── purge_poisonous_axiom.py
│   ├── regenerate_tools.py
│   ├── reproduce_dream_warning.py
│   ├── reproduce_filtering_failure.py
│   ├── test_*.py                # Various test modules
│   └── tests/                   # Nested test suite
│       ├── __init__.py
│       ├── probe_index*.py
│       ├── test_*.py            # Isolation and core tests
│       └── verify_*.py          # Verification tests
├── test_skills_env/             # Test skill environment
│   ├── __init__.py
│   └── test_skill.py
├── knowledge_base/             # Research and knowledge
│   └── research-reports/
│       └── rope_scaling*.json
├── CODEAGENCY/                 # Code agency evaluation
│   ├── evaluation.md
│   ├── diagnosis.md
│   ├── plan.md
│   └── trace_analysis.md
├── pyproject.toml              # Python project config
├── mcp_servers.json            # MCP server config
├── mcp_servers_example.json    # MCP server examples
├── README.md                   # Project readme
├── setup_env.sh                # Environment setup
├── start.sh                    # Start script
└── .venv/                      # Python virtual environment
```

## Naming Conventions

| Pattern | Example | Purpose |
|---------|---------|---------|
| `snake_case.py` | `graph_rlm/__init__.py` | Python module files |
| `PascalCase` | `TestAxiomRelevance` | Python classes |
| `snake_case()` | `def query()` | Python functions |
| `SCREAMING_SNAKE_CASE` | `MAX_DEPTH` | Python constants |
| `kebab-case.jsx` | (React components) | React component files |

## Key File Locations

### Core Entry Points

| File | Purpose |
|------|---------|
| `graph_rlm/__init__.py` | Main RLM exports |
| `graph_rlm/backend/main.py` | FastAPI application |

### Configuration

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project metadata, dependencies |
| `graph_rlm/frontend/package.json` | Frontend dependencies |
| `.trunk/configs/ruff.toml` | Linter configuration |
| `.vscode/settings.json` | Editor settings |

### Database & MCP

| File | Purpose |
|------|---------|
| `mcp_servers.json` | MCP server definitions |
| `mcp_servers_example.json` | Example MCP configs |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | Project overview |
| `CODEAGENCY/` | Evaluation and diagnosis docs |
| `knowledge_base/` | Research reports |

## Module Organization

### By Function

| Directory | Contents | Purpose |
|-----------|----------|---------|
| `graph_rlm/` | Core RLM modules | Main application |
| `graph_rlm/backend/` | API handlers | REST endpoints |
| `graph_rlm/frontend/` | React app | User interface |
| `scripts/` | CLI tools | Command-line utilities |
| `skills/` | MCP tools | External integrations |
| `tests/` | Test suite | Verification |

### By Responsibility

| Pattern | Example | Count |
|---------|---------|-------|
| Test modules | `test_*.py` | ~40+ files |
| Skills | `*.py` in `skills/` | ~50+ files |
| Nested tests | `tests/tests/` | ~15 files |

## Critical Paths

### Query Path

```
User -> frontend/ -> graph_rlm/backend/main.py -> graph_rlm/__init__.py -> RLM.query() -> FalkorDB
```

### Ingestion Path

```
Document -> ingest_document() -> Miner -> Coder -> Axiom Library -> Sheaf Monitor
```

### Skill Path

```
MCP Tool Call -> skills/*.py -> External Service -> Response
```

## Source Code Locations

| Type | Location |
|------|-----------|
| Python core | `graph_rlm/` |
| Python scripts | `scripts/` |
| Python skills | `skills/` |
| Python tests | `tests/` |
| Frontend | `graph_rlm/frontend/` |
| Configuration | Root directory |

## Configuration File Locations

| File | Type | Purpose |
|------|------|---------|
| `pyproject.toml` | TOML | Python project |
| `package.json` | JSON | Node.js project |
| `mcp_servers.json` | JSON | MCP servers |
| `ruff.toml` | TOML | Linter |
| `settings.json` | JSON | Editor |