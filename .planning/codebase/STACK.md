# STACK.md

## Languages

| Language | Version | Purpose |
|----------|---------|---------|
| Python | >= 3.13 | Core logic, agents, CAG system |
| TypeScript | ~5.9.3 | Frontend UI |
| JSON | - | Configuration files |

## Runtime Environments

- **Python**: Native CPython with `uv` for package management
- **Node.js**: Vite development server for frontend
- **Docker**: Containerized MCP servers and database

## Core Frameworks

### Python Backend

| Framework | Version | Purpose |
|-----------|---------|---------|
| FastAPI | >= 0.128.0 | Web framework for REST API |
| FalkorDB | >= 1.4.0 | Graph + Vector database for memory |
| LangChain | >= 1.2.7 | LLM orchestration |
| LangChain Community | >= 0.4.1 | Additional LLM integrations |
| Pydantic Settings | >= 2.12.0 | Configuration management |
| Uvicorn | >= 0.40.0 | ASGI server |

### AI & LLM Integration

| Library | Version | Purpose |
|---------|---------|---------|
| OpenAI | >= 2.15.0 | LLM provider |
| OpenRouter | >= 0.1.3 | LLM aggregator (xAI Grok, etc.) |
| Ollama | >= 0.6.1 | Local LLM runtime |
| Tiktoken | >= 0.12.0 | Token counting |

### Data & Processing

| Library | Version | Purpose |
|---------|---------|---------|
| NetworkX | >= 3.6.1 | Graph algorithms |
| NumPy | >= 2.4.1 | Numerical computing |
| Pandas | >= 2.0.0 | Data manipulation |
| SciPy | >= 1.17.0 | Scientific computing |

### MCP (Model Context Protocol)

| Library | Version | Purpose |
|---------|---------|---------|
| MCP | >= 1.26.0 | Tool/skill integration framework |

### Utilities

| Library | Version | Purpose |
|---------|---------|---------|
| Pydantic | - | Data validation (via pydantic-settings) |
| termcolor | >= 2.4.0 | Colored terminal output |
| nest-asyncio | >= 1.6.0 | Async event loop nesting |
| psutil | >= 7.2.2 | System monitoring |
| Unidecode | >= 1.4.0 | Unicode text normalization |
| Pillow | >= 12.1.0 | Image processing |
| PyMuPDF | >= 1.26.7 | PDF parsing |

### Development & Testing

| Library | Version | Purpose |
|---------|---------|---------|
| pytest | >= 9.0.2 | Test framework |
| pytest-asyncio | >= 1.3.0 | Async test support |

### Frontend (TypeScript)

| Library | Version | Purpose |
|---------|---------|---------|
| React | ^19.2.0 | UI framework |
| React DOM | ^19.2.0 | DOM rendering |
| React Router DOM | ^7.11.0 | Client-side routing |
| Framer Motion | ^12.23.26 | Animations |
| Recharts | ^3.6.0 | Data visualization |
| Lucide React | ^0.562.0 | Icons |
| Axios | ^1.13.2 | HTTP client |
| Tailwind CSS | ^4.1.18 | Utility-first CSS |

### Frontend Build Tools

| Tool | Version | Purpose |
|------|---------|---------|
| Vite | ^7.2.4 | Build tool and dev server |
| TypeScript | ~5.9.3 | Type checking |
| ESLint | ^9.39.1 | Code linting |
| PostCSS | ^8.5.6 | CSS processing |

## Package Managers

| Manager | Purpose |
|---------|---------|
| `uv` | Python package management (primary) |
| npm | Node.js package management |
| Docker | Container management |

## Configuration Files

- `pyproject.toml` - Python project configuration
- `graph_rlm/frontend/package.json` - Frontend dependencies
- `.trunk/configs/` - Linter configuration (ruff, yamllint)
- `.vscode/settings.json` - Editor settings
- `mcp_servers.json` - MCP server configuration
- `mcp_servers_example.json` - MCP server examples

## Key Dependencies

```
falkordb>=1.4.0          # Graph + Vector database
fastapi>=0.128.0        # Web framework
langchain>=1.2.7        # LLM orchestration
networkx>=3.6.1         # Graph algorithms
numpy>=2.4.1           # Numerical computing
openai>=2.15.0          # LLM provider
openrouter>=0.1.3      # LLM aggregator
uvicorn[standard]>=0.40.0  # ASGI server
mcp>=1.26.0            # Model Context Protocol
pytest>=9.0.2          # Testing
```

## Runtime Requirements

- Python 3.13 or higher
- Node.js (for frontend development)
- Docker (for MCP servers and FalkorDB)
- FalkorDB database (graph and vector storage)

## Environment Variables

See `.env` file (not committed) for:
- LLM API keys (OpenAI, OpenRouter)
- Database connection strings
- MCP server configurations