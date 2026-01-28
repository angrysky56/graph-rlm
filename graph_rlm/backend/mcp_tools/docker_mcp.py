"""
Auto-generated wrapper for docker-mcp MCP server.

This module provides Python function wrappers for all tools
exposed by the docker-mcp server.

Do not edit manually.
"""

from typing import Any


def create_container(image: str, name: str | None = None, ports: dict[str, Any] | None = None, environment: dict[str, Any] | None = None) -> Any:
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

    # Build parameters dict, excluding None values
    params = {}
    if image is not None:
        params["image"] = image
    if name is not None:
        params["name"] = name
    if ports is not None:
        params["ports"] = ports
    if environment is not None:
        params["environment"] = environment


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="docker-mcp",
            tool_name="create-container",
            arguments=params,
        )
    return asyncio.run(_async_call())


def deploy_compose(compose_yaml: str, project_name: str) -> Any:
    """Deploy a Docker Compose stack

    Args:
        compose_yaml: 
        project_name: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if compose_yaml is not None:
        params["compose_yaml"] = compose_yaml
    if project_name is not None:
        params["project_name"] = project_name


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="docker-mcp",
            tool_name="deploy-compose",
            arguments=params,
        )
    return asyncio.run(_async_call())


def get_logs(container_name: str) -> Any:
    """Retrieve the latest logs for a specified Docker container

    Args:
        container_name: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if container_name is not None:
        params["container_name"] = container_name


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="docker-mcp",
            tool_name="get-logs",
            arguments=params,
        )
    return asyncio.run(_async_call())


def list_containers() -> Any:
    """List all Docker containers

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="docker-mcp",
            tool_name="list-containers",
            arguments=params,
        )
    return asyncio.run(_async_call())



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['create-container', 'deploy-compose', 'get-logs', 'list-containers']
