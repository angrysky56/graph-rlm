"""
Auto-generated wrapper for wolframalpha MCP server.

This module provides Python function wrappers for all tools
exposed by the wolframalpha server.

Do not edit manually.
"""

from typing import Any


def ask_llm(query: str | Any = None, **kwargs) -> Any:
    """Ask WolframAlpha a query and get LLM-optimized structured response with multiple formats

    Args:
        query: The query to ask WolframAlpha

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if query is not None:
        mcp_args["query"] = query

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="wolframalpha",
            tool_name="ask_llm",
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


def get_simple_answer(query: str | Any = None, **kwargs) -> Any:
    """Get a simplified, LLM-friendly answer focusing on the most relevant information

    Args:
        query: The query to ask WolframAlpha

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if query is not None:
        mcp_args["query"] = query

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="wolframalpha",
            tool_name="get_simple_answer",
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


def validate_key(**kwargs) -> Any:
    """Validate the WolframAlpha LLM API key

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="wolframalpha",
            tool_name="validate_key",
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
    return ['ask_llm', 'get_simple_answer', 'validate_key']
