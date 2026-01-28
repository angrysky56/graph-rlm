"""
Auto-generated wrapper for verifier-graph MCP server.

This module provides Python function wrappers for all tools
exposed by the verifier-graph server.

Do not edit manually.
"""

from typing import Any


def propose_thought(type: str, content: str, parentIds: list[str] | None = None, edgeTypes: list[str] | None = None) -> Any:
    """Propose a new thought node to the reasoning graph. The Graph Kernel (∂) will validate constraints before committing.
        
Node Types:
- PREMISE: Axiom, fact, or retrieved data (can be root)
- WARRANT: Intermediate reasoning step
- CLAIM: Conclusion or assertion
- TOOL_CALL: Request to execute external function
- TOOL_RESULT: Output from tool (requires TOOL_CALL parent)
- CONSTRAINT: System rule
- REBUTTAL: Counter-argument

Constraints enforced:
- Orphan Prevention: Non-root nodes must have parents
- Tool Causality: TOOL_RESULT requires TOOL_CALL parent
- Acyclicity: Graph must remain a DAG

    Args:
        type: The type of thought node
        content: The content/text of the thought
        parentIds: IDs of parent nodes this thought derives from
        edgeTypes: Edge types for each parent (optional, defaults to DERIVED_FROM)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if type is not None:
        params["type"] = type
    if content is not None:
        params["content"] = content
    if parentIds is not None:
        params["parentIds"] = parentIds
    if edgeTypes is not None:
        params["edgeTypes"] = edgeTypes


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="verifier-graph",
            tool_name="propose_thought",
            arguments=params,
        )
    return asyncio.run(_async_call())


def get_context(nodeId: str, maxDepth: float | None = None) -> Any:
    """Retrieve the causal ancestors of a node - the 'causal light cone' that should be loaded for reasoning about this node.

    Args:
        nodeId: ID of the node to get context for
        maxDepth: Maximum depth to traverse (default: 10)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if nodeId is not None:
        params["nodeId"] = nodeId
    if maxDepth is not None:
        params["maxDepth"] = maxDepth


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="verifier-graph",
            tool_name="get_context",
            arguments=params,
        )
    return asyncio.run(_async_call())


def get_reasoning_chain(claimId: str) -> Any:
    """Get the full provenance path from root to a specific claim. Shows exactly which premises and reasoning led to this conclusion.

    Args:
        claimId: ID of the claim node to trace

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if claimId is not None:
        params["claimId"] = claimId


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="verifier-graph",
            tool_name="get_reasoning_chain",
            arguments=params,
        )
    return asyncio.run(_async_call())


def query_graph(query: str, nodeType: str | None = None) -> Any:
    """Search for nodes by content. Returns matching valid nodes.

    Args:
        query: Text to search for in node content
        nodeType: Filter by node type (optional)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if query is not None:
        params["query"] = query
    if nodeType is not None:
        params["nodeType"] = nodeType


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="verifier-graph",
            tool_name="query_graph",
            arguments=params,
        )
    return asyncio.run(_async_call())


def get_graph_state() -> Any:
    """Get the complete current state of the reasoning graph, including all nodes, edges, and metadata.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="verifier-graph",
            tool_name="get_graph_state",
            arguments=params,
        )
    return asyncio.run(_async_call())


def get_node(nodeId: str) -> Any:
    """Get a specific node by ID.

    Args:
        nodeId: ID of the node to retrieve

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if nodeId is not None:
        params["nodeId"] = nodeId


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="verifier-graph",
            tool_name="get_node",
            arguments=params,
        )
    return asyncio.run(_async_call())


def clear_graph() -> Any:
    """Reset the reasoning graph. Use with caution - all nodes and edges will be deleted.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="verifier-graph",
            tool_name="clear_graph",
            arguments=params,
        )
    return asyncio.run(_async_call())



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['propose_thought', 'get_context', 'get_reasoning_chain', 'query_graph', 'get_graph_state', 'get_node', 'clear_graph']
