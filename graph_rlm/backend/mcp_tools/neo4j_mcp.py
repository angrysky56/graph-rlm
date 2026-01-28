"""
Auto-generated wrapper for neo4j-mcp MCP server.

This module provides Python function wrappers for all tools
exposed by the neo4j-mcp server.

Do not edit manually.
"""

from typing import Any


def get_schema() -> Any:
    """
		Retrieve the schema information from the Neo4j database, including node labels, relationship types, and property keys.
		If the database contains no data, no schema information is returned.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="neo4j-mcp",
            tool_name="get-schema",
            arguments=params,
        )
    return asyncio.run(_async_call())


def list_gds_procedures() -> Any:
    """Use this tool to discover what graph science and analytics functions are available in the current Neo4j environment. It returns a structured list describing each function — what it does, how to use it, the inputs it needs, and what kind of results it produces. Do this before any reasoning, query generation, or analysis so you know what capabilities exist. Graph science and analytics functions help you with centrality, community detection, similarity, path finding, and identifying dependencies between nodes. The tool helps you understand the analytical capabilities of the system so that you can plan or compose the right graph science operations automatically. An empty response indicates that GDS is not installed and the user should be told to install it. Remember to use unique names for graph data science projections to avoid collisions and to drop them afterwards to save memory. You must always tell the user the function you will use.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="neo4j-mcp",
            tool_name="list-gds-procedures",
            arguments=params,
        )
    return asyncio.run(_async_call())


def read_cypher(query: str, params: dict[str, Any]) -> Any:
    """read-cypher can run only read-only Cypher statements. For write operations (CREATE, MERGE, DELETE, SET, etc...), schema/admin commands, or PROFILE queries, use write-cypher instead.

    Args:
        query: The Cypher query to execute
        params: Parameters to pass to the Cypher query

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if query is not None:
        params["query"] = query
    if params is not None:
        params["params"] = params


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="neo4j-mcp",
            tool_name="read-cypher",
            arguments=params,
        )
    return asyncio.run(_async_call())


def write_cypher(query: str, params: dict[str, Any]) -> Any:
    """write-cypher executes any arbitrary Cypher query, with write access, against the user-configured Neo4j database.

    Args:
        query: The Cypher query to execute
        params: Parameters to pass to the Cypher query

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if query is not None:
        params["query"] = query
    if params is not None:
        params["params"] = params


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="neo4j-mcp",
            tool_name="write-cypher",
            arguments=params,
        )
    return asyncio.run(_async_call())



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['get-schema', 'list-gds-procedures', 'read-cypher', 'write-cypher']
