# INTEGRATIONS.md

## Database Systems

### FalkorDB (Primary)

| Aspect | Details |
|--------|---------|
| Type | Graph + Vector database |
| Version | >= 1.4.0 |
| Purpose | Persistent graph memory and vector storage |
| Connection | Docker container or local instance |
| Used by | `graph_rlm/` core modules |

**Key Features:**
- Graph node/edge storage for "Graph of Thoughts" (GoT)
- Vector indexing for semantic similarity search
- Persistent session state across recursive calls

### SQLite (Optional)

Used in test environments for ephemeral state.

## LLM Providers

### OpenAI

| Aspect | Details |
|--------|---------|
| API | `openai>=2.15.0` |
| Purpose | Primary LLM provider |
| Models | GPT-4, GPT-3.5 Turbo, etc. |
| Configuration | Via `OPENAI_API_KEY` environment variable |

### OpenRouter

| Aspect | Details |
|--------|---------|
| Library | `openrouter>=0.1.3` |
| Purpose | Unified API for multiple LLMs |
| Supported Models | xAI Grok, Anthropic, Google, etc. |
| Configuration | Via `OPENROUTER_API_KEY` environment variable |

**Advantages:**
- Single API key for multiple providers
- Fallback capabilities
- Cost optimization

### Ollama (Local)

| Aspect | Details |
|--------|---------|
| Library | `ollama>=0.6.1` |
| Purpose | Local LLM runtime |
| Models | Any Ollama-supported model |
| Configuration | Via `OLLAMA_HOST` environment variable |

**Use Cases:**
- Privacy-sensitive applications
- Development and testing
- Offline operation

## MCP (Model Context Protocol) Integration

### MCP Server Configuration

| File | Purpose |
|------|---------|
| `mcp_servers.json` | Active MCP server configuration |
| `mcp_servers_example.json` | Example configurations |

### Available MCP Servers

Located in `skills/` directory:
- Various skill modules implement MCP tools
- Dynamic tool loading via MCP protocol
- See `skills/` for available integrations

**Key MCP Features:**
- Tool/skill discovery and loading
- Streaming tool calls
- Resource management

## Web APIs

### FastAPI Backend

| Aspect | Details |
|--------|---------|
| Framework | FastAPI >= 0.128.0 |
| Server | Uvicorn >= 0.40.0 |
| Port | 8000 (default) |
| Purpose | REST API for frontend and external clients |

**Key Endpoints (examples):**
- Graph operations (nodes, edges, queries)
- Recursive language model queries (`rlm.query()`)
- Document ingestion (`ingest_document()`)
- CAG axiomatic system endpoints

### Frontend API

| Aspect | Details |
|--------|---------|
| Framework | React + Vite |
| Port | 5173 (default) |
| Purpose | Interactive UI for graph visualization |

**Key Features:**
- Live graph visualization with D3.js
- Recursive query interface
- Axiomatic monitoring dashboard

## Document Processing

### PDF Extraction

| Library | Purpose |
|---------|---------|
| PyMuPDF >= 1.26.7 | PDF text and metadata extraction |

### Image Processing

| Library | Purpose |
|---------|---------|
| Pillow >= 12.1.0 | Image loading and processing |
| ascii-magic >= 2.7.4 | ASCII art generation |

## External Services

### Search and Research

| Service | Library | Purpose |
|---------|---------|---------|
| Web Search | `skills/search_web.py` | General web search |
| Wikipedia | `skills/wiki_extract.py` | Wikipedia article extraction |
| Wolfram | `skills/wolfram_query.py` | Computational knowledge |

### Code and Research

| Service | Purpose |
|---------|---------|
| arXiv | Research paper search (via skills) |
| GitHub | Code search and repository access |

## Configuration Integration Points

### Environment Variables

```bash
# LLM Providers
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-...
OLLAMA_HOST=http://localhost:11434

# Database
FALKORDB_HOST=localhost
FALKORDB_PORT=6379

# MCP Servers
MCP_SERVER_CONFIG=mcp_servers.json
```

### Configuration Files

| File | Format | Purpose |
|------|--------|---------|
| `pyproject.toml` | TOML | Python project config |
| `.env` | Plain text | Environment variables |
| `mcp_servers.json` | JSON | MCP server definitions |
| `.vscode/settings.json` | JSON | Editor configuration |

## Data Flow

```
Frontend (React) <--> FastAPI Backend <--> FalkorDB
                                              ^
                                              |
                                       MCP Servers (Docker)
```

## Monitoring & Observability

### structlog

| Aspect | Details |
|--------|---------|
| Library | `structlog>=24.2.0` |
| Purpose | Structured JSON logging |
| Output | Console (colored) and File (JSON) |
| Used by | All backend modules via `logger.py` |

**Key Features:**
- Contextual logging (correlation IDs, session IDs)
- JSON formatting for ELK/log aggregation
- Async-aware logging
- Thread-local context propagation

### Pydantic AI

| Aspect | Details |
|--------|---------|
| Library | `pydantic-ai>=1.59.0` |
| Purpose | Agentic AI with Pydantic validation |
| Used by | Advanced agent features and structured generation |

## Authentication

- API key-based authentication for LLM providers
- No built-in authentication for local development
- MCP server authentication via environment variables