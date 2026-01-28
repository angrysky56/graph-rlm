"""
Auto-generated wrapper for gitmcp MCP server.

This module provides Python function wrappers for all tools
exposed by the gitmcp server.

Do not edit manually.
"""

from typing import Any


def match_common_libs_owner_repo_mapping(library: str) -> Any:
    """Match a library name to an owner/repo. Don't use it if you have an owner and repo already. Use this first if only a library name was provided. If found - you can use owner and repo to call other tools. If not found - try to use the library name directly in other tools.

    Args:
        library: The name of the library to try and match to an owner/repo.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if library is not None:
        params["library"] = library


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="gitmcp",
            tool_name="match_common_libs_owner_repo_mapping",
            arguments=params,
        )
    return asyncio.run(_async_call())


def fetch_generic_documentation(owner: str, repo: str) -> Any:
    """Fetch documentation for any GitHub repository by providing owner and project name

    Args:
        owner: The GitHub repository owner (username or organization)
        repo: The GitHub repository name

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if owner is not None:
        params["owner"] = owner
    if repo is not None:
        params["repo"] = repo


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="gitmcp",
            tool_name="fetch_generic_documentation",
            arguments=params,
        )
    return asyncio.run(_async_call())


def search_generic_documentation(owner: str, repo: str, query: str) -> Any:
    """Semantically search in documentation for any GitHub repository by providing owner, project name, and search query. Useful for specific queries.

    Args:
        owner: The GitHub repository owner (username or organization)
        repo: The GitHub repository name
        query: The search query to find relevant documentation

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if owner is not None:
        params["owner"] = owner
    if repo is not None:
        params["repo"] = repo
    if query is not None:
        params["query"] = query


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="gitmcp",
            tool_name="search_generic_documentation",
            arguments=params,
        )
    return asyncio.run(_async_call())


def search_generic_code(owner: str, repo: str, query: str, page: float | None = None) -> Any:
    """Search for code in any GitHub repository by providing owner, project name, and search query. Returns matching files. Supports pagination with 30 results per page.

    Args:
        owner: The GitHub repository owner (username or organization)
        repo: The GitHub repository name
        query: The search query to find relevant code files
        page: Page number to retrieve (starting from 1). Each page contains 30 results.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if owner is not None:
        params["owner"] = owner
    if repo is not None:
        params["repo"] = repo
    if query is not None:
        params["query"] = query
    if page is not None:
        params["page"] = page


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="gitmcp",
            tool_name="search_generic_code",
            arguments=params,
        )
    return asyncio.run(_async_call())


def fetch_generic_url_content(url: str) -> Any:
    """Generic tool to fetch content from any absolute URL, respecting robots.txt rules. Use this to retrieve referenced urls (absolute urls) that were mentioned in previously fetched documentation.

    Args:
        url: The URL of the document or page to fetch

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if url is not None:
        params["url"] = url


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="gitmcp",
            tool_name="fetch_generic_url_content",
            arguments=params,
        )
    return asyncio.run(_async_call())



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['match_common_libs_owner_repo_mapping', 'fetch_generic_documentation', 'search_generic_documentation', 'search_generic_code', 'fetch_generic_url_content']
