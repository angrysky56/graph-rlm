"""
Auto-generated wrapper for mcp-logic MCP server.

This module provides Python function wrappers for all tools
exposed by the mcp-logic server.

Do not edit manually.
"""

from typing import Any


def prove(premises: list[str], conclusion: str) -> Any:
    """Prove a logical statement using Prover9

    Args:
        premises: List of logical premises
        conclusion: Statement to prove

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if premises is not None:
        params["premises"] = premises
    if conclusion is not None:
        params["conclusion"] = conclusion


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-logic",
            tool_name="prove",
            arguments=params,
        )
    return asyncio.run(_async_call())


def check_well_formed(statements: list[str]) -> Any:
    """Check if logical statements are well-formed with detailed syntax validation

    Args:
        statements: Logical statements to check

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if statements is not None:
        params["statements"] = statements


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-logic",
            tool_name="check-well-formed",
            arguments=params,
        )
    return asyncio.run(_async_call())


def find_model(premises: list[str], domain_size: int | None = None) -> Any:
    """Use Mace4 to find a finite model satisfying the given premises

    Args:
        premises: List of logical premises
        domain_size: Optional: specific domain size to search (default: incrementally search 2-10)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if premises is not None:
        params["premises"] = premises
    if domain_size is not None:
        params["domain_size"] = domain_size


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-logic",
            tool_name="find-model",
            arguments=params,
        )
    return asyncio.run(_async_call())


def find_counterexample(premises: list[str], conclusion: str, domain_size: int | None = None) -> Any:
    """Use Mace4 to find a counterexample showing the conclusion doesn't follow from premises

    Args:
        premises: List of logical premises
        conclusion: Conclusion to disprove
        domain_size: Optional: specific domain size to search

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if premises is not None:
        params["premises"] = premises
    if conclusion is not None:
        params["conclusion"] = conclusion
    if domain_size is not None:
        params["domain_size"] = domain_size


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-logic",
            tool_name="find-counterexample",
            arguments=params,
        )
    return asyncio.run(_async_call())


def verify_commutativity(path_a: list[str], path_b: list[str], object_start: str, object_end: str, with_category_axioms: bool | None = None) -> Any:
    """Verify that a categorical diagram commutes by generating FOL premises and conclusion

    Args:
        path_a: List of morphism names in first path
        path_b: List of morphism names in second path
        object_start: Starting object
        object_end: Ending object
        with_category_axioms: Include basic category theory axioms (default: true)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if path_a is not None:
        params["path_a"] = path_a
    if path_b is not None:
        params["path_b"] = path_b
    if object_start is not None:
        params["object_start"] = object_start
    if object_end is not None:
        params["object_end"] = object_end
    if with_category_axioms is not None:
        params["with_category_axioms"] = with_category_axioms


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-logic",
            tool_name="verify-commutativity",
            arguments=params,
        )
    return asyncio.run(_async_call())


def get_category_axioms(concept: str, functor_name: str | None = None) -> Any:
    """Get FOL axioms for category theory concepts (category, functor, natural transformation)

    Args:
        concept: Which concept's axioms to retrieve
        functor_name: For functor axioms: name of the functor (default: F)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if concept is not None:
        params["concept"] = concept
    if functor_name is not None:
        params["functor_name"] = functor_name


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-logic",
            tool_name="get-category-axioms",
            arguments=params,
        )
    return asyncio.run(_async_call())



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['prove', 'check-well-formed', 'find-model', 'find-counterexample', 'verify-commutativity', 'get-category-axioms']
