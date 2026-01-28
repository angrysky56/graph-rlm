"""
Auto-generated wrapper for brave-search MCP server.

This module provides Python function wrappers for all tools
exposed by the brave-search server.

Do not edit manually.
"""

from typing import Any


def brave_web_search(query: str, count: float | None = None, offset: float | None = None) -> Any:
    """Performs a web search using the Brave Search API, ideal for general queries, news, articles, and online content. Use this for broad information gathering, recent events, or when you need diverse web sources. Supports pagination, content filtering, and freshness controls. Maximum 20 results per request, with offset for pagination. 

    Args:
        query: Search query (max 400 chars, 50 words)
        count: Number of results (1-20, default 10)
        offset: Pagination offset (max 9, default 0)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if query is not None:
        params["query"] = query
    if count is not None:
        params["count"] = count
    if offset is not None:
        params["offset"] = offset


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="brave-search",
            tool_name="brave_web_search",
            arguments=params,
        )
    return asyncio.run(_async_call())


def brave_local_search(query: str, count: float | None = None) -> Any:
    """Searches for local businesses and places using Brave's Local Search API. Best for queries related to physical locations, businesses, restaurants, services, etc. Returns detailed information including:
- Business names and addresses
- Ratings and review counts
- Phone numbers and opening hours
Use this when the query implies 'near me' or mentions specific locations. Automatically falls back to web search if no local results are found.

    Args:
        query: Local search query (e.g. 'pizza near Central Park')
        count: Number of results (1-20, default 5)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if query is not None:
        params["query"] = query
    if count is not None:
        params["count"] = count


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="brave-search",
            tool_name="brave_local_search",
            arguments=params,
        )
    return asyncio.run(_async_call())



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['brave_web_search', 'brave_local_search']
