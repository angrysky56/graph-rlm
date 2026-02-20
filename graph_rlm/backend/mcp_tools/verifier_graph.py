"""
Auto-generated wrapper for verifier-graph MCP server.

This module provides Python function wrappers for all tools
exposed by the verifier-graph server.

Do not edit manually.
"""

from typing import Any


def propose_thought(thoughtType: str | Any = None, content: str | Any = None, parentIds: list[str] | None = None, edgeTypes: list[str] | None = None, **kwargs) -> Any:
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
        thoughtType: The type of thought node
        content: The content/text of the thought
        parentIds: IDs of parent nodes this thought derives from
        edgeTypes: Edge types for each parent (optional, defaults to DERIVED_FROM)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    # Resilience: Handle aliases for 'thoughtType'
    actual_thoughtType = thoughtType or kwargs.get('type') or kwargs.get('node_type') or kwargs.get('thought_type')
    if actual_thoughtType is not None:
        mcp_args["thoughtType"] = actual_thoughtType
    if content is not None:
        mcp_args["content"] = content
    if parentIds is not None:
        mcp_args["parentIds"] = parentIds
    if edgeTypes is not None:
        mcp_args["edgeTypes"] = edgeTypes

    async def _async_call():
        return await call_mcp_tool(
            server_name="verifier-graph",
            tool_name="propose_thought",
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


def get_context(nodeId: str | Any = None, maxDepth: float | None = None, **kwargs) -> Any:
    """Retrieve the causal ancestors of a node - the 'causal light cone' that should be loaded for reasoning about this node.

    Args:
        nodeId: ID of the node to get context for
        maxDepth: Maximum depth to traverse (default: 10)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if nodeId is not None:
        mcp_args["nodeId"] = nodeId
    if maxDepth is not None:
        mcp_args["maxDepth"] = maxDepth

    async def _async_call():
        return await call_mcp_tool(
            server_name="verifier-graph",
            tool_name="get_context",
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


def get_reasoning_chain(claimId: str | Any = None, **kwargs) -> Any:
    """Get the full provenance path from root to a specific claim. Shows exactly which premises and reasoning led to this conclusion.

    Args:
        claimId: ID of the claim node to trace

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if claimId is not None:
        mcp_args["claimId"] = claimId

    async def _async_call():
        return await call_mcp_tool(
            server_name="verifier-graph",
            tool_name="get_reasoning_chain",
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


def query_graph(query: str | Any = None, nodeType: str | None = None, **kwargs) -> Any:
    """Search for nodes by content. Returns matching valid nodes.

    Args:
        query: Text to search for in node content
        nodeType: Filter by node type (optional)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if query is not None:
        mcp_args["query"] = query
    if nodeType is not None:
        mcp_args["nodeType"] = nodeType

    async def _async_call():
        return await call_mcp_tool(
            server_name="verifier-graph",
            tool_name="query_graph",
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


def get_graph_state(**kwargs) -> Any:
    """Get the complete current state of the reasoning graph, including all nodes, edges, and metadata.

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
            server_name="verifier-graph",
            tool_name="get_graph_state",
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


def get_node(nodeId: str | Any = None, **kwargs) -> Any:
    """Get a specific node by ID.

    Args:
        nodeId: ID of the node to retrieve

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if nodeId is not None:
        mcp_args["nodeId"] = nodeId

    async def _async_call():
        return await call_mcp_tool(
            server_name="verifier-graph",
            tool_name="get_node",
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


def clear_graph(**kwargs) -> Any:
    """Reset the reasoning graph. Use with caution - all nodes and edges will be deleted.

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
            server_name="verifier-graph",
            tool_name="clear_graph",
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
    return ['propose_thought', 'get_context', 'get_reasoning_chain', 'query_graph', 'get_graph_state', 'get_node', 'clear_graph']
