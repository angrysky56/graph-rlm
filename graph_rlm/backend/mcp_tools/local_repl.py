"""
Auto-generated wrapper for local-repl MCP server.

This module provides Python function wrappers for all tools
exposed by the local-repl server.

Do not edit manually.
"""

from typing import Any


def create_python_repl() -> Any:
    """
    Create a new Python REPL environment.
    
    Returns:
        str: ID of the new REPL
    

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="local-repl",
            tool_name="create_python_repl",
            arguments=params,
        )
    return asyncio.run(_async_call())


def run_python_in_repl(code: str, repl_id: str) -> Any:
    """
    Execute Python code in a REPL.
    
    Args:
        code: Python code to execute
        repl_id: ID of the REPL to use
        
    Returns:
        str: Result of the execution including stdout, stderr, and the return value
    

    Args:
        code: 
        repl_id: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if code is not None:
        params["code"] = code
    if repl_id is not None:
        params["repl_id"] = repl_id


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="local-repl",
            tool_name="run_python_in_repl",
            arguments=params,
        )
    return asyncio.run(_async_call())


def list_active_repls() -> Any:
    """List all active REPL instances and their IDs.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="local-repl",
            tool_name="list_active_repls",
            arguments=params,
        )
    return asyncio.run(_async_call())


def get_repl_info(repl_id: str) -> Any:
    """
    Get information about a specific REPL instance.
    
    Args:
        repl_id: ID of the REPL to get info for
        
    Returns:
        str: Information about the REPL
    

    Args:
        repl_id: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if repl_id is not None:
        params["repl_id"] = repl_id


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="local-repl",
            tool_name="get_repl_info",
            arguments=params,
        )
    return asyncio.run(_async_call())


def delete_repl(repl_id: str) -> Any:
    """
    Delete a REPL instance.
    
    Args:
        repl_id: ID of the REPL to delete
        
    Returns:
        str: Confirmation message
    

    Args:
        repl_id: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if repl_id is not None:
        params["repl_id"] = repl_id


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="local-repl",
            tool_name="delete_repl",
            arguments=params,
        )
    return asyncio.run(_async_call())


def initialize_modular_empowerment(repl_id: str) -> Any:
    """
    Initialize the Modular Empowerment Framework in a specific REPL.

    Args:
        repl_id: ID of the REPL to initialize in

    Returns:
        str: Result of the initialization
    

    Args:
        repl_id: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if repl_id is not None:
        params["repl_id"] = repl_id


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="local-repl",
            tool_name="initialize_modular_empowerment",
            arguments=params,
        )
    return asyncio.run(_async_call())



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['create_python_repl', 'run_python_in_repl', 'list_active_repls', 'get_repl_info', 'delete_repl', 'initialize_modular_empowerment']
