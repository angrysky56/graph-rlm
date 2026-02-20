"""
Auto-generated wrapper for community-server-llm-txt MCP server.

This module provides Python function wrappers for all tools
exposed by the community-server-llm-txt server.

Do not edit manually.
"""

from typing import Any


def get_llm_txt(id: float | Any = None, page: float | None = None, **kwargs) -> Any:
    """Fetch an LLM.txt file from a given URL. Format your response in beautiful markdown.

    Args:
        id: The ID of the LLM.txt file to fetch. Must be obtained first using the list_llm_txt command.
        page: Page number to fetch, starting from 1. Each page contains a fixed number of characters.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if id is not None:
        mcp_args["id"] = id
    if page is not None:
        mcp_args["page"] = page

    async def _async_call():
        return await call_mcp_tool(
            server_name="community-server-llm-txt",
            tool_name="get_llm_txt",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def list_llm_txt(**kwargs) -> Any:
    """List available LLM.txt files from the directory. Use this first before fetching a specific LLM.txt file. Format your response in beautiful markdown.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}

    async def _async_call():
        return await call_mcp_tool(
            server_name="community-server-llm-txt",
            tool_name="list_llm_txt",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def search_llm_txt(id: float | Any = None, queries: list[str] | Any = None, context_lines: float | None = None, **kwargs) -> Any:
    """Search for multiple substrings in an LLM.txt file. Requires a valid ID obtained from list_llm_txt command. Returns snippets with page numbers for each match. Format your response in beautiful markdown, using code blocks for snippets.

    Args:
        id: The ID of the LLM.txt file to search in. Must be obtained first using the list_llm_txt command.
        queries: Array of substrings to search for. Each query is searched case-insensitively. At least one query is required.
        context_lines: Number of lines to show before and after each match for context. Defaults to 2 lines.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if id is not None:
        mcp_args["id"] = id
    if queries is not None:
        mcp_args["queries"] = queries
    if context_lines is not None:
        mcp_args["context_lines"] = context_lines

    async def _async_call():
        return await call_mcp_tool(
            server_name="community-server-llm-txt",
            tool_name="search_llm_txt",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['get_llm_txt', 'list_llm_txt', 'search_llm_txt']
