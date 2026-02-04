# MCP Tools Quick Reference

## Access Pattern
```python
await mcp.<MODULE_NAME>.<FUNCTION_NAME>(args...)
```

## Available Modules & Tools

### Web Search
- **Module**: `brave_search`
- **Function**: `brave_web_search(query, count=None, offset=None)`
- **Usage**: `await mcp.brave_search.brave_web_search(query="...")`

### Web Scraping
- **Module**: `mcp_server_firecrawl`
- **Functions**: `firecrawl_scrape()`, `firecrawl_extract()`, `firecrawl_crawl()`, `firecrawl_batch_scrape()`
- **Usage**: `await mcp.mcp_server_firecrawl.firecrawl_scrape(url="...")`

### Academic Papers
- **Module**: `arxiv_mcp_server`
- **Function**: `arxiv_query(query, max_results=None)`
- **Usage**: `await mcp.arxiv_mcp_server.arxiv_query(query="...")`

### Docker Operations
- **Module**: `docker_mcp`
- **Functions**: `docker_ps()`, `docker_logs()`, `docker_exec()`, etc.
- **Usage**: `await mcp.docker_mcp.docker_ps()`

### Database (Neo4j)
- **Module**: `neo4j_mcp`
- **Functions**: `neo4j_query()`, `neo4j_transaction()`, etc.
- **Usage**: `await mcp.neo4j_mcp.neo4j_query(cypher="...")`

### Browser Automation
- **Module**: `playwright`
- **Functions**: `playwright_navigate()`, `playwright_click()`, etc.
- **Usage**: `await mcp.playwright.playwright_navigate(url="...")`

## To List All Available Modules
```python
print(dir(mcp))
```

## To List Tools in a Module
```python
print(dir(mcp.brave_search))
```

## To See Tool Documentation
```python
print(mcp.brave_search.brave_web_search.__doc__)
```

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `AttributeError: No MCP server found with name 'brave'` | Using old alias | Use `mcp.brave_search` instead |
| `MCP Server 'X' has no tool 'Y'` | Wrong tool name or typo | Check `dir(mcp.<module>)` |
| `<coroutine object ...>` in output | Not awaiting the call | Add `await` before the call |
| `ModuleNotFoundError` | Module not available | Check it exists in `dir(mcp)` |

## Example: Research Pipeline

```python
# 1. Search the web
results = await mcp.brave_search.brave_web_search(query="machine learning transformers")

# 2. Scrape the top result
content = await mcp.mcp_server_firecrawl.firecrawl_scrape(
    url=results[0]['url'],
    formats=['markdown']
)

# 3. Search academic papers
papers = await mcp.arxiv_mcp_server.arxiv_query(
    query="transformer architecture neural networks",
    max_results=5
)

# Done!
rlm.done("Research complete")
```

## Writing Skills That Use MCP

```python
async def research_skill(topic: str):
    """A skill that researches a topic using multiple MCP tools."""
    from graph_rlm.backend.src.core.agent import mcp

    # Search the web
    results = await mcp.brave_search.brave_web_search(query=topic)

    # Scrape top result
    if results:
        content = await mcp.mcp_server_firecrawl.firecrawl_scrape(
            url=results[0].get('url'),
            formats=['markdown']
        )
        return f"Topic: {topic}\n\nContent:\n{content}"

    return f"No results found for {topic}"
```

## See Also

- [MCP_TOOL_USAGE.md](MCP_TOOL_USAGE.md) - Complete guide with troubleshooting
- [MCP_FIX_SUMMARY.md](MCP_FIX_SUMMARY.md) - History and rationale of the fix
