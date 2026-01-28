"""
Auto-generated wrapper for mcp-obsidian MCP server.

This module provides Python function wrappers for all tools
exposed by the mcp-obsidian server.

Do not edit manually.
"""

from typing import Any


def obsidian_list_files_in_dir(dirpath: str) -> Any:
    """Lists all files and directories that exist in a specific Obsidian directory.

    Args:
        dirpath: Path to list files from (relative to your vault root). Note that empty directories will not be returned.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if dirpath is not None:
        params["dirpath"] = dirpath


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-obsidian",
            tool_name="obsidian_list_files_in_dir",
            arguments=params,
        )
    return asyncio.run(_async_call())


def obsidian_list_files_in_vault() -> Any:
    """Lists all files and directories in the root directory of your Obsidian vault.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-obsidian",
            tool_name="obsidian_list_files_in_vault",
            arguments=params,
        )
    return asyncio.run(_async_call())


def obsidian_get_file_contents(filepath: str) -> Any:
    """Return the content of a single file in your vault.

    Args:
        filepath: Path to the relevant file (relative to your vault root).

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if filepath is not None:
        params["filepath"] = filepath


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-obsidian",
            tool_name="obsidian_get_file_contents",
            arguments=params,
        )
    return asyncio.run(_async_call())


def obsidian_simple_search(query: str, context_length: int | None = None) -> Any:
    """Simple search for documents matching a specified text query across all files in the vault. 
            Use this tool when you want to do a simple text search

    Args:
        query: Text to a simple search for in the vault.
        context_length: How much context to return around the matching string (default: 100)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if query is not None:
        params["query"] = query
    if context_length is not None:
        params["context_length"] = context_length


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-obsidian",
            tool_name="obsidian_simple_search",
            arguments=params,
        )
    return asyncio.run(_async_call())


def obsidian_patch_content(filepath: str, operation: str, target_type: str, target: str, content: str) -> Any:
    """Insert content into an existing note relative to a heading, block reference, or frontmatter field.

    Args:
        filepath: Path to the file (relative to vault root)
        operation: Operation to perform (append, prepend, or replace)
        target_type: Type of target to patch
        target: Target identifier (heading path, block reference, or frontmatter field)
        content: Content to insert

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if filepath is not None:
        params["filepath"] = filepath
    if operation is not None:
        params["operation"] = operation
    if target_type is not None:
        params["target_type"] = target_type
    if target is not None:
        params["target"] = target
    if content is not None:
        params["content"] = content


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-obsidian",
            tool_name="obsidian_patch_content",
            arguments=params,
        )
    return asyncio.run(_async_call())


def obsidian_append_content(filepath: str, content: str) -> Any:
    """Append content to a new or existing file in the vault.

    Args:
        filepath: Path to the file (relative to vault root)
        content: Content to append to the file

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if filepath is not None:
        params["filepath"] = filepath
    if content is not None:
        params["content"] = content


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-obsidian",
            tool_name="obsidian_append_content",
            arguments=params,
        )
    return asyncio.run(_async_call())


def obsidian_delete_file(filepath: str, confirm: bool) -> Any:
    """Delete a file or directory from the vault.

    Args:
        filepath: Path to the file or directory to delete (relative to vault root)
        confirm: Confirmation to delete the file (must be true)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if filepath is not None:
        params["filepath"] = filepath
    if confirm is not None:
        params["confirm"] = confirm


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-obsidian",
            tool_name="obsidian_delete_file",
            arguments=params,
        )
    return asyncio.run(_async_call())


def obsidian_complex_search(query: dict[str, Any]) -> Any:
    """Complex search for documents using a JsonLogic query. 
           Supports standard JsonLogic operators plus 'glob' and 'regexp' for pattern matching. Results must be non-falsy.

           Use this tool when you want to do a complex search, e.g. for all documents with certain tags etc.
           

    Args:
        query: JsonLogic query object. Example: {"glob": ["*.md", {"var": "path"}]} matches all markdown files

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if query is not None:
        params["query"] = query


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-obsidian",
            tool_name="obsidian_complex_search",
            arguments=params,
        )
    return asyncio.run(_async_call())


def obsidian_batch_get_file_contents(filepaths: list[str]) -> Any:
    """Return the contents of multiple files in your vault, concatenated with headers.

    Args:
        filepaths: List of file paths to read

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if filepaths is not None:
        params["filepaths"] = filepaths


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-obsidian",
            tool_name="obsidian_batch_get_file_contents",
            arguments=params,
        )
    return asyncio.run(_async_call())


def obsidian_get_periodic_note(period: str) -> Any:
    """Get current periodic note for the specified period.

    Args:
        period: The period type (daily, weekly, monthly, quarterly, yearly)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if period is not None:
        params["period"] = period


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-obsidian",
            tool_name="obsidian_get_periodic_note",
            arguments=params,
        )
    return asyncio.run(_async_call())


def obsidian_get_recent_periodic_notes(period: str, limit: int | None = None, include_content: bool | None = None) -> Any:
    """Get most recent periodic notes for the specified period type.

    Args:
        period: The period type (daily, weekly, monthly, quarterly, yearly)
        limit: Maximum number of notes to return (default: 5)
        include_content: Whether to include note content (default: false)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if period is not None:
        params["period"] = period
    if limit is not None:
        params["limit"] = limit
    if include_content is not None:
        params["include_content"] = include_content


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-obsidian",
            tool_name="obsidian_get_recent_periodic_notes",
            arguments=params,
        )
    return asyncio.run(_async_call())


def obsidian_get_recent_changes(limit: int | None = None, days: int | None = None) -> Any:
    """Get recently modified files in the vault.

    Args:
        limit: Maximum number of files to return (default: 10)
        days: Only include files modified within this many days (default: 90)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if limit is not None:
        params["limit"] = limit
    if days is not None:
        params["days"] = days


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-obsidian",
            tool_name="obsidian_get_recent_changes",
            arguments=params,
        )
    return asyncio.run(_async_call())



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['obsidian_list_files_in_dir', 'obsidian_list_files_in_vault', 'obsidian_get_file_contents', 'obsidian_simple_search', 'obsidian_patch_content', 'obsidian_append_content', 'obsidian_delete_file', 'obsidian_complex_search', 'obsidian_batch_get_file_contents', 'obsidian_get_periodic_note', 'obsidian_get_recent_periodic_notes', 'obsidian_get_recent_changes']
