"""
Auto-generated wrapper for wolframalpha MCP server.

This module provides Python function wrappers for all tools
exposed by the wolframalpha server.

Do not edit manually.
"""

from typing import Any


def ask_llm(query: str) -> Any:
    """Ask WolframAlpha a query and get LLM-optimized structured response with multiple formats

    Args:
        query: The query to ask WolframAlpha

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if query is not None:
        params["query"] = query


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="wolframalpha",
            tool_name="ask_llm",
            arguments=params,
        )
    return asyncio.run(_async_call())


def get_simple_answer(query: str) -> Any:
    """Get a simplified, LLM-friendly answer focusing on the most relevant information

    Args:
        query: The query to ask WolframAlpha

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if query is not None:
        params["query"] = query


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="wolframalpha",
            tool_name="get_simple_answer",
            arguments=params,
        )
    return asyncio.run(_async_call())


def validate_key() -> Any:
    """Validate the WolframAlpha LLM API key

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="wolframalpha",
            tool_name="validate_key",
            arguments=params,
        )
    return asyncio.run(_async_call())



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['ask_llm', 'get_simple_answer', 'validate_key']
