"""
Auto-generated wrapper for gitmcp MCP server.

This module provides Python function wrappers for all tools
exposed by the gitmcp server.

Do not edit manually.
"""

from typing import Any


def match_common_libs_owner_repo_mapping(library: str | Any = None, **kwargs) -> Any:
    """Match a library name to an owner/repo. Don't use it if you have an owner and repo already. Use this first if only a library name was provided. If found - you can use owner and repo to call other tools. If not found - try to use the library name directly in other tools.

    Args:
        library: The name of the library to try and match to an owner/repo.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if library is not None:
        mcp_args["library"] = library

    async def _async_call():
        return await call_mcp_tool(
            server_name="gitmcp",
            tool_name="match_common_libs_owner_repo_mapping",
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


def fetch_generic_documentation(owner: str | Any = None, repo: str | Any = None, **kwargs) -> Any:
    """Fetch documentation for any GitHub repository by providing owner and project name

    Args:
        owner: The GitHub repository owner (username or organization)
        repo: The GitHub repository name

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if owner is not None:
        mcp_args["owner"] = owner
    if repo is not None:
        mcp_args["repo"] = repo

    async def _async_call():
        return await call_mcp_tool(
            server_name="gitmcp",
            tool_name="fetch_generic_documentation",
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


def search_generic_documentation(owner: str | Any = None, repo: str | Any = None, query: str | Any = None, **kwargs) -> Any:
    """Semantically search in documentation for any GitHub repository by providing owner, project name, and search query. Useful for specific queries.

    Args:
        owner: The GitHub repository owner (username or organization)
        repo: The GitHub repository name
        query: The search query to find relevant documentation

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if owner is not None:
        mcp_args["owner"] = owner
    if repo is not None:
        mcp_args["repo"] = repo
    if query is not None:
        mcp_args["query"] = query

    async def _async_call():
        return await call_mcp_tool(
            server_name="gitmcp",
            tool_name="search_generic_documentation",
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


def search_generic_code(owner: str | Any = None, repo: str | Any = None, query: str | Any = None, page: float | None = None, **kwargs) -> Any:
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
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if owner is not None:
        mcp_args["owner"] = owner
    if repo is not None:
        mcp_args["repo"] = repo
    if query is not None:
        mcp_args["query"] = query
    if page is not None:
        mcp_args["page"] = page

    async def _async_call():
        return await call_mcp_tool(
            server_name="gitmcp",
            tool_name="search_generic_code",
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


def fetch_generic_url_content(url: str | Any = None, **kwargs) -> Any:
    """Generic tool to fetch content from any absolute URL, respecting robots.txt rules. Use this to retrieve referenced urls (absolute urls) that were mentioned in previously fetched documentation.

    Args:
        url: The URL of the document or page to fetch

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if url is not None:
        mcp_args["url"] = url

    async def _async_call():
        return await call_mcp_tool(
            server_name="gitmcp",
            tool_name="fetch_generic_url_content",
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
    return ['match_common_libs_owner_repo_mapping', 'fetch_generic_documentation', 'search_generic_documentation', 'search_generic_code', 'fetch_generic_url_content']
