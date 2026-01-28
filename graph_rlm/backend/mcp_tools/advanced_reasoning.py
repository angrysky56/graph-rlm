"""
Auto-generated wrapper for advanced-reasoning MCP server.

This module provides Python function wrappers for all tools
exposed by the advanced-reasoning server.

Do not edit manually.
"""

from typing import Any


def advanced_reasoning(thought: str, nextThoughtNeeded: bool, thoughtNumber: int, totalThoughts: int, confidence: float | None = None, reasoning_quality: str | None = None, meta_thought: str | None = None, goal: str | None = None, progress: float | None = None, hypothesis: str | None = None, test_plan: str | None = None, test_result: str | None = None, evidence: list[str] | None = None, session_id: str | None = None, builds_on: list[str] | None = None, challenges: list[str] | None = None, isRevision: bool | None = None, revisesThought: int | None = None, branchFromThought: int | None = None, branchId: str | None = None, needsMoreThoughts: bool | None = None) -> Any:
    """Advanced cognitive reasoning tool that builds on sequential thinking with meta-cognition, hypothesis testing, and integrated memory.

Key Features:
- Meta-cognitive assessment and confidence tracking
- Hypothesis formulation and testing capabilities
- Integrated graph-based memory system
- Dynamic reasoning quality evaluation
- Session-based context management
- Evidence tracking and validation

Enhanced Parameters:
- thought: Your reasoning step (required)
- thoughtNumber/totalThoughts: Sequential tracking (required)
- nextThoughtNeeded: Continue flag (required)
- confidence: Self-assessment 0.0-1.0 (default: 0.5)
- reasoning_quality: 'low'|'medium'|'high' (default: 'medium')
- meta_thought: Reflection on your reasoning process
- hypothesis: Current working hypothesis
- test_plan: How to validate the hypothesis
- test_result: Outcome of testing
- evidence: Supporting/contradicting evidence
- session_id: Link to reasoning session
- goal: Overall objective
- progress: 0.0-1.0 completion estimate

Branching (inherited from sequential thinking):
- isRevision/revisesThought: Revise previous thoughts
- branchFromThought/branchId: Explore alternatives

Use this tool for complex reasoning that benefits from:
- Self-reflection and confidence tracking
- Systematic hypothesis development
- Memory of previous insights
- Quality assessment of reasoning

    Args:
        thought: Your current reasoning step
        nextThoughtNeeded: Whether another thought step is needed
        thoughtNumber: Current thought number
        totalThoughts: Estimated total thoughts needed
        confidence: Confidence in this reasoning step (0.0-1.0)
        reasoning_quality: Assessment of reasoning quality
        meta_thought: Meta-cognitive reflection on your reasoning process
        goal: Overall goal or objective
        progress: Progress toward goal (0.0-1.0)
        hypothesis: Current working hypothesis
        test_plan: Plan for testing the hypothesis
        test_result: Result of hypothesis testing
        evidence: Evidence for/against hypothesis
        session_id: Reasoning session identifier
        builds_on: Previous thoughts this builds on
        challenges: Ideas this challenges or contradicts
        isRevision: Whether this revises previous thinking
        revisesThought: Which thought is being reconsidered
        branchFromThought: Branching point thought number
        branchId: Branch identifier
        needsMoreThoughts: If more thoughts are needed

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if thought is not None:
        params["thought"] = thought
    if nextThoughtNeeded is not None:
        params["nextThoughtNeeded"] = nextThoughtNeeded
    if thoughtNumber is not None:
        params["thoughtNumber"] = thoughtNumber
    if totalThoughts is not None:
        params["totalThoughts"] = totalThoughts
    if confidence is not None:
        params["confidence"] = confidence
    if reasoning_quality is not None:
        params["reasoning_quality"] = reasoning_quality
    if meta_thought is not None:
        params["meta_thought"] = meta_thought
    if goal is not None:
        params["goal"] = goal
    if progress is not None:
        params["progress"] = progress
    if hypothesis is not None:
        params["hypothesis"] = hypothesis
    if test_plan is not None:
        params["test_plan"] = test_plan
    if test_result is not None:
        params["test_result"] = test_result
    if evidence is not None:
        params["evidence"] = evidence
    if session_id is not None:
        params["session_id"] = session_id
    if builds_on is not None:
        params["builds_on"] = builds_on
    if challenges is not None:
        params["challenges"] = challenges
    if isRevision is not None:
        params["isRevision"] = isRevision
    if revisesThought is not None:
        params["revisesThought"] = revisesThought
    if branchFromThought is not None:
        params["branchFromThought"] = branchFromThought
    if branchId is not None:
        params["branchId"] = branchId
    if needsMoreThoughts is not None:
        params["needsMoreThoughts"] = needsMoreThoughts


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="advanced-reasoning",
            tool_name="advanced_reasoning",
            arguments=params,
        )
    return asyncio.run(_async_call())


def query_reasoning_memory(session_id: str, query: str) -> Any:
    """Query the integrated memory system to find related insights, hypotheses, and evidence.

Useful for:
- Finding similar problems solved before
- Retrieving relevant hypotheses and evidence
- Understanding connections between ideas
- Building on previous reasoning sessions

Parameters:
- session_id: The reasoning session to query within (required)
- query: What to search for in memory (required)

Returns related memories with confidence scores and connection information.

    Args:
        session_id: Reasoning session identifier
        query: What to search for in memory

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if session_id is not None:
        params["session_id"] = session_id
    if query is not None:
        params["query"] = query


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="advanced-reasoning",
            tool_name="query_reasoning_memory",
            arguments=params,
        )
    return asyncio.run(_async_call())


def create_memory_library(library_name: str) -> Any:
    """Create a new named memory library for organized knowledge storage.

Enables you to create separate, named memory libraries for different projects, domains, or contexts.
Library names must contain only letters, numbers, underscores, and hyphens.

Parameters:
- library_name: Name for the new library (required)

Returns success status and message.

    Args:
        library_name: Name for the new memory library

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if library_name is not None:
        params["library_name"] = library_name


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="advanced-reasoning",
            tool_name="create_memory_library",
            arguments=params,
        )
    return asyncio.run(_async_call())


def list_memory_libraries() -> Any:
    """List all available memory libraries with metadata.

Shows all existing memory libraries with information about:
- Library name
- Number of memory nodes
- Last modified date

Returns organized, searchable library information.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="advanced-reasoning",
            tool_name="list_memory_libraries",
            arguments=params,
        )
    return asyncio.run(_async_call())


def switch_memory_library(library_name: str) -> Any:
    """Switch to a different memory library.

Allows you to switch between different memory libraries for different contexts or projects.
Current session state is saved before switching.

Parameters:
- library_name: Name of the library to switch to (required)

Returns success status and message.

    Args:
        library_name: Name of the library to switch to

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if library_name is not None:
        params["library_name"] = library_name


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="advanced-reasoning",
            tool_name="switch_memory_library",
            arguments=params,
        )
    return asyncio.run(_async_call())


def get_current_library_info() -> Any:
    """Get information about the currently active memory library.

Shows current library name, number of nodes, sessions, and other metadata.

Returns current library information.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="advanced-reasoning",
            tool_name="get_current_library_info",
            arguments=params,
        )
    return asyncio.run(_async_call())


def create_system_json(name: str, domain: str, description: str, data: dict[str, Any], tags: list[str] | None = None) -> Any:
    """Create a new system JSON file for storing coherent detailed searchable data or instructions and workflows for any domain or action.

Parameters:
- name: Name for the system JSON file (required) - alphanumeric, underscore, hyphen only
- domain: Domain or category for the data (required)
- description: Description of what this system JSON contains (required)
- data: The structured data to store (required) - can be any JSON-serializable object
- tags: Optional array of tags for searchability

Returns success status and confirmation message.

    Args:
        name: Name for the system JSON file (alphanumeric, underscore, hyphen only)
        domain: Domain or category for the data
        description: Description of what this system JSON contains
        data: The structured data to store
        tags: Optional array of tags for searchability

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if name is not None:
        params["name"] = name
    if domain is not None:
        params["domain"] = domain
    if description is not None:
        params["description"] = description
    if data is not None:
        params["data"] = data
    if tags is not None:
        params["tags"] = tags


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="advanced-reasoning",
            tool_name="create_system_json",
            arguments=params,
        )
    return asyncio.run(_async_call())


def get_system_json(name: str) -> Any:
    """Retrieve a system JSON file by name.

Parameters:
- name: Name of the system JSON file to retrieve (required)

Returns the complete system JSON data including metadata and content.

    Args:
        name: Name of the system JSON file to retrieve

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if name is not None:
        params["name"] = name


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="advanced-reasoning",
            tool_name="get_system_json",
            arguments=params,
        )
    return asyncio.run(_async_call())


def search_system_json(query: str) -> Any:
    """Search through system JSON files by query.

Parameters:
- query: Search query to find matching system JSON files (required)

Returns matching files with relevance scores.

    Args:
        query: Search query to find matching system JSON files

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
            server_name="advanced-reasoning",
            tool_name="search_system_json",
            arguments=params,
        )
    return asyncio.run(_async_call())


def list_system_json() -> Any:
    """List all available system JSON files.

Returns list of all system JSON files with their names, domains, and descriptions.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="advanced-reasoning",
            tool_name="list_system_json",
            arguments=params,
        )
    return asyncio.run(_async_call())



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['advanced_reasoning', 'query_reasoning_memory', 'create_memory_library', 'list_memory_libraries', 'switch_memory_library', 'get_current_library_info', 'create_system_json', 'get_system_json', 'search_system_json', 'list_system_json']
