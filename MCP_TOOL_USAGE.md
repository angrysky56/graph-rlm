# MCP Tool Usage Guide

## Overview

The Graph-RLM system uses Model Context Protocol (MCP) servers to expose tools to the agent. This guide explains how to correctly access and use these tools.

## Architecture

### How MCP Tools Work

1. **Tool Modules**: MCP servers are auto-generated as Python modules in `graph_rlm/backend/mcp_tools/`
2. **Lazy Loading**: Tools are lazily loaded when accessed to avoid startup delays
3. **REPL Integration**: Tools are exposed to the REPL via the `mcp` namespace
4. **Async Support**: All MCP tool calls are async-safe and wrapped automatically

### Key Fix (January 30, 2026)

**BREAKING**: The system previously supported aliases like `mcp.brave`, `mcp.firecrawl`, etc. These have been **removed** to prevent confusion and tool discovery failures.

**REASON**: Aliases caused mismatches between the namespace alias and the actual module name, preventing the tool discovery system from finding tools correctly. The error:

```
MCP Server 'mcp_firecrawl' has no tool 'scrape'
```

...occurred because the system was looking for `scrape` in the `mcp_firecrawl` (alias) but the actual module is `mcp_server_firecrawl`.

---

## Correct Tool Access Patterns

### Using Tools in REPL Code

All MCP tools are accessed via the actual module name:

```python
# Correct ✅
result = await mcp.brave_search.brave_web_search(query="...")
result = await mcp.mcp_server_firecrawl.firecrawl_scrape(url="...")
result = await mcp.arxiv_mcp_server.arxiv_query(...)

# Incorrect ❌
result = await mcp.brave.search(query="...")      # 'brave' alias no longer exists
result = await mcp.firecrawl.scrape(url="...")    # 'firecrawl' alias no longer exists
```

### Tool Naming Convention

Each module in `graph_rlm/backend/mcp_tools/` corresponds to an MCP server:

| Module Name | Tools Inside | Usage |
|---|---|---|
| `brave_search.py` | `brave_web_search()` | Web search via Brave |
| `mcp_server_firecrawl.py` | `firecrawl_scrape()`, `firecrawl_extract()`, etc. | Web scraping and content extraction |
| `arxiv_mcp_server.py` | `arxiv_query()` | ArXiv paper search |
| `docker_mcp.py` | `docker_*()` functions | Docker operations |
| `neo4j_mcp.py` | `neo4j_*()` functions | Neo4j operations |
| `playwright.py` | `playwright_*()` functions | Browser automation |

### Available MCP Servers

Run this in the REPL to see all available servers:

```python
print(dir(mcp))
```

This will display all loaded MCP server modules (actual names, not aliases).

---

## Practical Examples

### Web Search

```python
# Search the web using Brave Search
result = await mcp.brave_search.brave_web_search(
    query="machine learning transformer architecture",
    count=5
)
print(result)
```

### Web Scraping

```python
# Scrape a webpage using Firecrawl
result = await mcp.mcp_server_firecrawl.firecrawl_scrape(
    url="https://example.com",
    formats=["markdown"]  # or ["json", "html", etc.]
)
print(result)
```

### ArXiv Paper Search

```python
# Search for papers on arXiv
result = await mcp.arxiv_mcp_server.arxiv_query(
    query="neural networks",
    max_results=5
)
print(result)
```

---

## Troubleshooting

### Error: "MCP Server 'X' has no tool 'Y'"

**Cause**: Trying to use an alias or incorrect tool name.

**Fix**:
1. Check the actual module name in `graph_rlm/backend/mcp_tools/`
2. Use the full function name from the generated wrapper
3. Example: Instead of `mcp.brave.search()`, use `mcp.brave_search.brave_web_search()`

### Error: "No MCP server found with name or alias 'X'"

**Cause**: Module name typo or module doesn't exist.

**Fix**:
1. List all available modules: `print(dir(mcp))`
2. Check module exists in `graph_rlm/backend/mcp_tools/`
3. Verify spelling matches exactly

### Tool Returns a Coroutine

**Cause**: Not awaiting the async call.

**Fix**: Always `await` MCP tool calls:

```python
# Incorrect ❌
result = mcp.brave_search.brave_web_search(query="...")
print(result)  # Prints: <coroutine object ...>

# Correct ✅
result = await mcp.brave_search.brave_web_search(query="...")
print(result)  # Prints: actual search results
```

---

## Writing Skills That Use MCP Tools

When writing skills that call MCP tools, always use the full module name:

```python
async def my_skill():
    """Perform web research."""
    # Import at the top of the skill
    from graph_rlm.backend.src.core.agent import mcp

    # Call with full module.function pattern
    search_result = await mcp.brave_search.brave_web_search(
        query="topic of interest"
    )

    return search_result
```

**Note**: Skills run in their own async context, so you must import and use `mcp` correctly.

---

## Migration Guide

If you have existing code using the old alias pattern, update as follows:

| Old (Broken) | New (Correct) |
|---|---|
| `mcp.brave` | `mcp.brave_search` |
| `mcp.firecrawl` | `mcp.mcp_server_firecrawl` |
| `mcp.arxiv` | `mcp.arxiv_mcp_server` |
| `mcp.brave.search()` | `mcp.brave_search.brave_web_search()` |
| `mcp.firecrawl.scrape()` | `mcp.mcp_server_firecrawl.firecrawl_scrape()` |

---

## Implementation Details

### How Tool Discovery Works

1. **Module Scan**: `LazyMCPNamespace._scan()` enumerates modules in `mcp_tools/`
2. **MCPServerNamespace Creation**: Each module gets an `MCPServerNamespace` wrapper
3. **Function Import**: Functions are imported from the module and wrapped as async callables
4. **Tool Recording**: Each call is recorded for audit/debugging
5. **Lazy Loading**: Actual module loading happens on first tool access, not during scan

### Why Aliases Were Removed

The alias system had multiple problems:

1. **Mismatch**: Alias name ≠ actual module name → tool discovery fails
2. **Ambiguity**: `mcp.brave` could refer to `brave_search` → confusion
3. **Collision**: Multiple aliases for the same module → unpredictable behavior
4. **Non-standard**: Not how MCP clients typically work

The new system is **explicit and traceable**: use the actual module name always.

---

## See Also

- [agent.py - MCPServerNamespace](../graph_rlm/backend/src/core/agent.py#L53-L125): Implementation details
- [mcp_tools/](../graph_rlm/backend/mcp_tools/): Generated tool wrappers
- [generator.py](../graph_rlm/backend/src/mcp_integration/generator.py): How wrappers are generated
- [runtime.py](../graph_rlm/backend/src/mcp_integration/runtime.py): Async execution layer
