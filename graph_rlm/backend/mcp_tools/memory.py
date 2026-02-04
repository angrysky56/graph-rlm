"""
Auto-generated wrapper for memory MCP server.

This module provides Python function wrappers for all tools
exposed by the memory server.

Do not edit manually.
"""

from typing import Any


def create_entities(entities: list[dict[str, Any]]) -> Any:
    """Create multiple new entities in the knowledge graph

    Args:
        entities: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict, excluding None values
    mcp_args = {}
    if entities is not None:
        mcp_args["entities"] = entities

    async def _async_call():
        return await call_mcp_tool(
            server_name="memory",
            tool_name="create_entities",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return the coroutine
            return _async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return asyncio.run(_async_call())


def create_relations(relations: list[dict[str, Any]]) -> Any:
    """Create multiple new relations between entities in the knowledge graph. Relations should be in active voice

    Args:
        relations: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict, excluding None values
    mcp_args = {}
    if relations is not None:
        mcp_args["relations"] = relations

    async def _async_call():
        return await call_mcp_tool(
            server_name="memory",
            tool_name="create_relations",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return the coroutine
            return _async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return asyncio.run(_async_call())


def add_observations(observations: list[dict[str, Any]]) -> Any:
    """Add new observations to existing entities in the knowledge graph

    Args:
        observations: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict, excluding None values
    mcp_args = {}
    if observations is not None:
        mcp_args["observations"] = observations

    async def _async_call():
        return await call_mcp_tool(
            server_name="memory",
            tool_name="add_observations",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return the coroutine
            return _async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return asyncio.run(_async_call())


def delete_entities(entityNames: list[str]) -> Any:
    """Delete multiple entities and their associated relations from the knowledge graph

    Args:
        entityNames: An array of entity names to delete

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict, excluding None values
    mcp_args = {}
    if entityNames is not None:
        mcp_args["entityNames"] = entityNames

    async def _async_call():
        return await call_mcp_tool(
            server_name="memory",
            tool_name="delete_entities",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return the coroutine
            return _async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return asyncio.run(_async_call())


def delete_observations(deletions: list[dict[str, Any]]) -> Any:
    """Delete specific observations from entities in the knowledge graph

    Args:
        deletions: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict, excluding None values
    mcp_args = {}
    if deletions is not None:
        mcp_args["deletions"] = deletions

    async def _async_call():
        return await call_mcp_tool(
            server_name="memory",
            tool_name="delete_observations",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return the coroutine
            return _async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return asyncio.run(_async_call())


def delete_relations(relations: list[dict[str, Any]]) -> Any:
    """Delete multiple relations from the knowledge graph

    Args:
        relations: An array of relations to delete

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict, excluding None values
    mcp_args = {}
    if relations is not None:
        mcp_args["relations"] = relations

    async def _async_call():
        return await call_mcp_tool(
            server_name="memory",
            tool_name="delete_relations",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return the coroutine
            return _async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return asyncio.run(_async_call())


def read_graph() -> Any:
    """Read the entire knowledge graph

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict, excluding None values
    mcp_args = {}

    async def _async_call():
        return await call_mcp_tool(
            server_name="memory",
            tool_name="read_graph",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return the coroutine
            return _async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return asyncio.run(_async_call())


def search_nodes(query: str) -> Any:
    """Search for nodes in the knowledge graph based on a query

    Args:
        query: The search query to match against entity names, types, and observation content

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict, excluding None values
    mcp_args = {}
    if query is not None:
        mcp_args["query"] = query

    async def _async_call():
        return await call_mcp_tool(
            server_name="memory",
            tool_name="search_nodes",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return the coroutine
            return _async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return asyncio.run(_async_call())


def open_nodes(names: list[str]) -> Any:
    """Open specific nodes in the knowledge graph by their names

    Args:
        names: An array of entity names to retrieve

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict, excluding None values
    mcp_args = {}
    if names is not None:
        mcp_args["names"] = names

    async def _async_call():
        return await call_mcp_tool(
            server_name="memory",
            tool_name="open_nodes",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return the coroutine
            return _async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return asyncio.run(_async_call())



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['create_entities', 'create_relations', 'add_observations', 'delete_entities', 'delete_observations', 'delete_relations', 'read_graph', 'search_nodes', 'open_nodes']
