# Codebase Structure

**Analysis Date:** 2026-05-04

## Directory Layout

```
graph-rlm/
├── .github/                      # GitHub configuration
│   └── FUNDING.yml
├── .planning/                    # GSD planning artifacts
│   └── codebase/                 # This codebase map
├── .trunk/                       # Linter configuration
│   └── configs/
│       ├── .markdownlint.yaml
│       ├── ruff.toml
│       └── .yamllint.yaml
├── .vscode/                      # VS Code settings
│   └── settings.json
├── .venv/                        # Python virtual environment
├── docs/                         # Documentation
├── falkordb_data/                # FalkorDB data directory
├── graph_rlm/                    # Main package
│   ├── __init__.py               # RLM exports
│   ├── backend/                  # FastAPI backend
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI entry point
│   │   ├── rules.md              # Agent rules
│   │   ├── agent_venv/           # Isolated execution environment
│   │   ├── axioms_dir/           # Axiom validators (CAG)
│   │   │   ├── axiom_*.py        # Individual axioms
│   │   │   └── _disabled/        # Disabled axioms
│   │   ├── mcp_tools/            # Generated MCP tool wrappers
│   │   │   ├── mcp_*.py          # Server wrappers
│   │   │   └── __init__.py
│   │   ├── .python-version
│   │   └── src/                  # Core backend source
│   │       ├── __init__.py
│   │       ├── cli.py            # CLI interface
│   │       ├── core/             # Core modules
│   │       │   ├── __init__.py
│   │       │   ├── agent.py      # RLM Agent
│   │       │   ├── config.py     # Pydantic settings
│   │       │   ├── db.py         # FalkorDB client
│   │       │   ├── endpoints.py  # FastAPI endpoints
│   │       │   ├── guardrails.py # Guardrail system
│   │       │   ├── llm.py        # LLM interface
│   │       │   ├── logger.py     # Structured logging
│   │       │   ├── monitor.py    # Execution monitor
│   │       │   ├── navigator.py  # Curiosity-driven navigation
│   │       │   ├── omcd.py       # Metacognitive control
│   │       │   ├── prompts.py    # Prompt templates
│   │       │   ├── repe.py       # RepE safety monitor
│   │       │   ├── sheaf.py      # Sheaf topology monitor
│   │       │   ├── dream.py      # Dreamer consolidation
│   │       │   ├── state.py      # Execution state
│   │       │   ├── trace.py      # Trace utilities
│   │       │   ├── scratchpad_builder.py
│   │       │   ├── context_index.py
│   │       │   ├── rlm_interface.py
│   │       │   └── core.py
│   │       └── mcp_integration/  # MCP integration
│   │           ├── __init__.py
│   │           ├── config.py
│   │           ├── discovery.py  # Server discovery
│   │           ├── generator.py  # Tool generation
│   │           ├── runtime.py    # Process isolation
│   │           ├── kernel.py     # IPC kernel
│   │           ├── client.py     # MCP client
│   │           ├── skill_storage.py     # Skills/Axioms sync
│   │           ├── skill_harness.py
│   │           └── utils/
│   │               ├── schema_builder.py
│   │               └── task_schema.py
│   └── frontend/                 # React frontend
│       ├── package.json
│       ├── tsconfig.json
│       └── (React app source)
├── knowledge_base/               # Research and knowledge
│   ├── axioms/
│   ├── plans/
│   ├── research-reports/
│   ├── outputs/
│   └── workspace/
├── scripts/                      # Utility scripts
│   ├── generate_structure.py
│   ├── purge_orphaned_axioms.py
│   └── regenerate_tools.py
├── skills/                       # MCP skills (reusable)
│   ├── __init__.py
│   └── *.py                      # Individual skills
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── conftest.py               # Pytest fixtures (mock_registry, event_loop)
│   ├── test_*.py                 # Test modules
│   ├── verify_*.py               # Verification tests
│   ├── debug_*.py                # Debug utilities
│   ├── mocking/                  # Centralized mocking infrastructure
│   │   ├── falkordb.py           # FalkorDB mocks
│   │   ├── llm.py                # LLM service mocks
│   │   ├── external.py           # External API mocks
│   │   └── mocks.py              # MockRegistry class
│   └── unit/                     # Unit tests
├── test_skills_env/              # Test skill environment
├── CODEAGENCY/                   # Code agency evaluation
├── pyproject.toml                # Python project config
├── mcp_servers.json              # MCP server config
├── mcp_servers_example.json      # MCP server examples
├── MCP_TOOL_USAGE.md
├── README.md                     # Project readme
├── setup_env.sh                  # Environment setup
├── start.sh                      # Start script
├── .env                          # Environment variables
├── .env.example                  # Environment template
└── .gitignore
```

## Directory Purposes

**`graph_rlm/backend/src/core/`:**
- Purpose: Core backend modules
- Contains: Agent, database, LLM, endpoints, configuration, monitoring
- Key files: `agent.py`, `db.py`, `llm.py`, `config.py`, `endpoints.py`

**`graph_rlm/backend/src/mcp_integration/`:**
- Purpose: MCP protocol integration and skills management
- Contains: Server discovery, tool generation, skills/axioms sync, runtime isolation
- Key files: `skill_storage.py`, `runtime.py`, `discovery.py`, `generator.py`

**`graph_rlm/backend/axioms_dir/`:**
- Purpose: Constraint Augmented Generation validators
- Contains: Python functions that validate operations (e.g., `axiom_epistemic_victory_check.py`)
- Pattern: Functions named `axiom_*` that return bool or raise GuardrailError

**`graph_rlm/backend/mcp_tools/`:**
- Purpose: Generated MCP tool wrapper modules
- Contains: Auto-generated Python files from `mcp_servers.json` config
- Pattern: Files named after MCP servers (e.g., `playwright.py`, `brave_search.py`)

**`graph_rlm/frontend/`:**
- Purpose: React + Vite web UI
- Contains: Frontend source code, package.json, TypeScript configs
- Runs on: localhost:5173

**`skills/`:**
- Purpose: Reusable skill library
- Contains: Python modules with skill functions
- Pattern: Python files with `def skill_*()` or async functions

**`tests/`:**
- Purpose: Test suite and verification tools
- Contains: Test files, debug utilities, verification scripts
- Pattern: `test_*.py`, `verify_*.py`, `debug_*.py`

**`knowledge_base/`:**
- Purpose: Agent's persistent knowledge storage
- Contains: `axioms/`, `plans/`, `research-reports/`, `outputs/`, `workspace/`

## Key File Locations

**Entry Points:**
- `graph_rlm/backend/main.py`: FastAPI backend server
- `graph_rlm/backend/src/cli.py`: CLI interface
- `graph_rlm/__init__.py`: Package exports

**Configuration:**
- `graph_rlm/backend/src/core/config.py`: Pydantic settings (`.env` support)
- `pyproject.toml`: Python project metadata and dependencies
- `mcp_servers.json`: MCP server configurations

**Core Logic:**
- `graph_rlm/backend/src/core/agent.py`: Agent class (RLM core)
- `graph_rlm/backend/src/core/db.py`: FalkorDB graph client
- `graph_rlm/backend/src/core/llm.py`: LLM provider abstraction

**MCP Integration:**
- `graph_rlm/backend/src/mcp_integration/skill_storage.py`: Skills/Axioms sync
- `graph_rlm/backend/src/mcp_integration/runtime.py`: Isolated REPL via uv

**Monitoring:**
- `graph_rlm/backend/src/core/sheaf.py`: Sheaf topology monitor
- `graph_rlm/backend/src/core/repe.py`: RepE safety monitor
- `graph_rlm/backend/src/core/dream.py`: Dreamer consolidation

**Testing:**
- `tests/`: Test files directory
- `pytest.ini`: pytest configuration

## Naming Conventions

**Files:**
- `snake_case.py`: Python modules (e.g., `graph_rlm/backend/src/core/agent.py`)
- `PascalCase`: Python classes (e.g., `Agent`, `GraphClient`)
- `snake_case()`: Python functions and methods
- `SCREAMING_SNAKE_CASE`: Constants (e.g., `MAX_RECURSION_DEPTH`)
- `kebab-case.tsx`: React component files

**Directories:**
- `snake_case/`: Python package directories (e.g., `graph_rlm/backend/src/core/`)
- `lowercase/`: Configuration and utility directories (e.g., `scripts/`)

**Special Patterns:**
- `axiom_*.py`: Axiom validator files in `axioms_dir/`
- `mcp_*.py`: Generated MCP tool wrappers in `mcp_tools/`
- `test_*.py`: Test files in `tests/`
- `verify_*.py`: Verification scripts in `tests/`

## Where to Add New Code

**New Feature:**
- Primary code: `graph_rlm/backend/src/core/` (if core) or `skills/` (if reusable skill)
- Tests: `tests/test_feature_name.py`

**New MCP Server:**
- Configuration: Add to `mcp_servers.json`
- Generated wrapper: Auto-generated in `graph_rlm/backend/mcp_tools/`

**New Axiom (CAG):**
- Implementation: `graph_rlm/backend/axioms_dir/axiom_*.py`
- Pattern: Function taking context params, returning bool or raising GuardrailError

**New Backend Module:**
- Location: `graph_rlm/backend/src/core/` or `graph_rlm/backend/src/mcp_integration/`
- Export: Add to relevant `__init__.py`

**New Skill:**
- Implementation: `skills/skill_name.py`
- Pattern: Python file with exported function(s)

**Frontend Changes:**
- Components: `graph_rlm/frontend/src/components/`
- Pages: `graph_rlm/frontend/src/pages/`

## Special Directories

**`graph_rlm/backend/axioms_dir/_disabled/`:**
- Purpose: Deprecated or problematic axioms
- Generated: No
- Committed: Yes
- Note: Non-recursive sync ignores this directory

**`graph_rlm/backend/agent_venv/`:**
- Purpose: Isolated Python environment for code execution
- Generated: Yes (at runtime)
- Committed: No (in .gitignore)

**`falkordb_data/`:**
- Purpose: FalkorDB graph database storage
- Generated: Yes (by FalkorDB)
- Committed: No (in .gitignore)

**`.venv/`:**
- Purpose: Development virtual environment
- Generated: Yes (by uv/venv)
- Committed: No (in .gitignore)

---

*Structure analysis: 2026-02-12*
