"""
Auto-generated wrapper for docker-mcp MCP server.

This module provides Python function wrappers for all tools
exposed by the docker-mcp server.

Do not edit manually.
"""

from typing import Any


def create_container(image: str | Any = None, name: str | None = None, ports: dict[str, Any] | None = None, environment: dict[str, Any] | None = None, **kwargs) -> Any:
    """Create a new standalone Docker container

    Args:
        image: 
        name: 
        ports: 
        environment: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if image is not None:
        mcp_args["image"] = image
    if name is not None:
        mcp_args["name"] = name
    if ports is not None:
        mcp_args["ports"] = ports
    if environment is not None:
        mcp_args["environment"] = environment

    async def _async_call():
        return await call_mcp_tool(
            server_name="docker-mcp",
            tool_name="create-container",
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


def deploy_compose(compose_yaml: str | Any = None, project_name: str | Any = None, **kwargs) -> Any:
    """Deploy a Docker Compose stack

    Args:
        compose_yaml: 
        project_name: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if compose_yaml is not None:
        mcp_args["compose_yaml"] = compose_yaml
    if project_name is not None:
        mcp_args["project_name"] = project_name

    async def _async_call():
        return await call_mcp_tool(
            server_name="docker-mcp",
            tool_name="deploy-compose",
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


def get_logs(container_name: str | Any = None, **kwargs) -> Any:
    """Retrieve the latest logs for a specified Docker container

    Args:
        container_name: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if container_name is not None:
        mcp_args["container_name"] = container_name

    async def _async_call():
        return await call_mcp_tool(
            server_name="docker-mcp",
            tool_name="get-logs",
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


def list_containers(**kwargs) -> Any:
    """List all Docker containers

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
            server_name="docker-mcp",
            tool_name="list-containers",
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
    return ['create-container', 'deploy-compose', 'get-logs', 'list-containers']
