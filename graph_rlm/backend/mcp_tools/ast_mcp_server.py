"""
Auto-generated wrapper for ast-mcp-server MCP server.

This module provides Python function wrappers for all tools
exposed by the ast-mcp-server server.

Do not edit manually.
"""

from typing import Any


def parse_to_ast(code: Any | None = None, language: Any | None = None, filename: Any | None = None) -> Any:
    """Step 1: Parse code → AST (syntax tree). Use this to validate syntax or get a raw tree dump.

    Args:
        code: 
        language: 
        filename: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if code is not None:
        params["code"] = code
    if language is not None:
        params["language"] = language
    if filename is not None:
        params["filename"] = filename


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="parse_to_ast",
            arguments=params,
        )
    return asyncio.run(_async_call())


def generate_asg(code: Any | None = None, language: Any | None = None, filename: Any | None = None) -> Any:
    """Step 3: Parse code → AST → ASG (graph). Use this to explore basic relationships (edges) between nodes.

    Args:
        code: 
        language: 
        filename: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if code is not None:
        params["code"] = code
    if language is not None:
        params["language"] = language
    if filename is not None:
        params["filename"] = filename


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="generate_asg",
            arguments=params,
        )
    return asyncio.run(_async_call())


def analyze_code(code: Any | None = None, language: Any | None = None, filename: Any | None = None) -> Any:
    """Step 2: Extract metadata (Functions, Classes, Imports). Use this for high-level file summaries.

    Args:
        code: 
        language: 
        filename: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if code is not None:
        params["code"] = code
    if language is not None:
        params["language"] = language
    if filename is not None:
        params["filename"] = filename


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="analyze_code",
            arguments=params,
        )
    return asyncio.run(_async_call())


def parse_to_ast_incremental(code: Any | None = None, old_code: Any | None = None, language: Any | None = None, filename: Any | None = None) -> Any:
    """Step 1 (Enhanced): Incremental parsing. Use this instead of `parse_to_ast` for large files or edits.

    Args:
        code: 
        old_code: 
        language: 
        filename: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if code is not None:
        params["code"] = code
    if old_code is not None:
        params["old_code"] = old_code
    if language is not None:
        params["language"] = language
    if filename is not None:
        params["filename"] = filename


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="parse_to_ast_incremental",
            arguments=params,
        )
    return asyncio.run(_async_call())


def generate_enhanced_asg(code: Any | None = None, language: Any | None = None, filename: Any | None = None) -> Any:
    """Step 3 (Enhanced): Deep semantic analysis (Scope, Data Flow). Use for refactoring or complex queries.

    Args:
        code: 
        language: 
        filename: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if code is not None:
        params["code"] = code
    if language is not None:
        params["language"] = language
    if filename is not None:
        params["filename"] = filename


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="generate_enhanced_asg",
            arguments=params,
        )
    return asyncio.run(_async_call())


def diff_ast(old_code: str, new_code: str, language: Any | None = None, filename: Any | None = None) -> Any:
    """Compare two code versions semantically. Returns AST differences (nodes added/removed/changed).

    Args:
        old_code: 
        new_code: 
        language: 
        filename: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if old_code is not None:
        params["old_code"] = old_code
    if new_code is not None:
        params["new_code"] = new_code
    if language is not None:
        params["language"] = language
    if filename is not None:
        params["filename"] = filename


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="diff_ast",
            arguments=params,
        )
    return asyncio.run(_async_call())


def find_node_at_position(code: Any | None = None, line: int | None = None, column: int | None = None, language: Any | None = None, filename: Any | None = None) -> Any:
    """Interactive: Get AST node at a specific cursor line/column. Use for cursor-based context.

    Args:
        code: 
        line: 
        column: 
        language: 
        filename: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if code is not None:
        params["code"] = code
    if line is not None:
        params["line"] = line
    if column is not None:
        params["column"] = column
    if language is not None:
        params["language"] = language
    if filename is not None:
        params["filename"] = filename


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="find_node_at_position",
            arguments=params,
        )
    return asyncio.run(_async_call())


def search_code_patterns(code: str, pattern: str, language: Any | None = None, filename: Any | None = None) -> Any:
    """Search for structural patterns in code using ast-grep. Returns {matches, count}.

    Args:
        code: 
        pattern: 
        language: 
        filename: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if code is not None:
        params["code"] = code
    if pattern is not None:
        params["pattern"] = pattern
    if language is not None:
        params["language"] = language
    if filename is not None:
        params["filename"] = filename


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="search_code_patterns",
            arguments=params,
        )
    return asyncio.run(_async_call())


def transform_code_patterns(code: str, pattern: str, replacement: str, language: Any | None = None, filename: Any | None = None, preview_only: bool | None = None) -> Any:
    """Replace structural patterns in code using ast-grep. Returns {transformed_code, changes_applied}.

    Args:
        code: 
        pattern: 
        replacement: 
        language: 
        filename: 
        preview_only: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if code is not None:
        params["code"] = code
    if pattern is not None:
        params["pattern"] = pattern
    if replacement is not None:
        params["replacement"] = replacement
    if language is not None:
        params["language"] = language
    if filename is not None:
        params["filename"] = filename
    if preview_only is not None:
        params["preview_only"] = preview_only


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="transform_code_patterns",
            arguments=params,
        )
    return asyncio.run(_async_call())


def validate_ast_pattern(pattern: str, language: str) -> Any:
    """Check if ast-grep pattern syntax is valid for the specified language.

    Args:
        pattern: 
        language: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if pattern is not None:
        params["pattern"] = pattern
    if language is not None:
        params["language"] = language


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="validate_ast_pattern",
            arguments=params,
        )
    return asyncio.run(_async_call())


def list_transformation_examples() -> Any:
    """Get common ast-grep pattern examples for code modernization and refactoring.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="list_transformation_examples",
            arguments=params,
        )
    return asyncio.run(_async_call())


def sync_file_to_graph(code: str, file_path: str, language: Any | None = None) -> Any:
    """Parse code → store AST+ASG+metrics in Neo4j. Returns {stored: {ast_id, asg_id, analysis_id}}.

    Args:
        code: 
        file_path: 
        language: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if code is not None:
        params["code"] = code
    if file_path is not None:
        params["file_path"] = file_path
    if language is not None:
        params["language"] = language


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="sync_file_to_graph",
            arguments=params,
        )
    return asyncio.run(_async_call())


def query_neo4j_graph(query: str, parameters: Any | None = None) -> Any:
    """Execute Cypher query on code graph. Returns {records, count}.

    Args:
        query: 
        parameters: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if query is not None:
        params["query"] = query
    if parameters is not None:
        params["parameters"] = parameters


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="query_neo4j_graph",
            arguments=params,
        )
    return asyncio.run(_async_call())


def ask_uss_agent(query: str) -> Any:
    """Graph Query: Ask natural language questions about the codebase (uses Neo4j/ChromaDB).

    Args:
        query: 

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
            server_name="ast-mcp-server",
            tool_name="ask_uss_agent",
            arguments=params,
        )
    return asyncio.run(_async_call())


def uss_agent_status() -> Any:
    """Check status of the USS Agent services (Neo4j, ChromaDB, LLM).

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="uss_agent_status",
            arguments=params,
        )
    return asyncio.run(_async_call())


def analyze_source_file(project_name: str, code: Any | None = None, language: Any | None = None, filename: Any | None = None, include_summary: bool | None = None) -> Any:
    """Analyze a single source file, save reports to disk, and optionally generate an AI summary.

    Args:
        project_name: 
        code: 
        language: 
        filename: 
        include_summary: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if project_name is not None:
        params["project_name"] = project_name
    if code is not None:
        params["code"] = code
    if language is not None:
        params["language"] = language
    if filename is not None:
        params["filename"] = filename
    if include_summary is not None:
        params["include_summary"] = include_summary


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="analyze_source_file",
            arguments=params,
        )
    return asyncio.run(_async_call())


def analyze_project(project_path: str, project_name: str, file_extensions: Any | None = None, sync_to_db: bool | None = None, include_summary: bool | None = None) -> Any:
    """Recursively analyze a project, generate reports, and optionaly sync to Graph DB.

Args:
    project_path: Root directory to analyze
    project_name: Name of the project (for output grouping)
    file_extensions: List of extensions to include (default: .py, .js, .ts, .tsx, .go)
    sync_to_db: Whether to sync nodes/edges to Neo4j (default: True)
    include_summary: Whether to generate AI summaries for each file (default: True)


    Args:
        project_path: 
        project_name: 
        file_extensions: 
        sync_to_db: 
        include_summary: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if project_path is not None:
        params["project_path"] = project_path
    if project_name is not None:
        params["project_name"] = project_name
    if file_extensions is not None:
        params["file_extensions"] = file_extensions
    if sync_to_db is not None:
        params["sync_to_db"] = sync_to_db
    if include_summary is not None:
        params["include_summary"] = include_summary


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="analyze_project",
            arguments=params,
        )
    return asyncio.run(_async_call())



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['parse_to_ast', 'generate_asg', 'analyze_code', 'parse_to_ast_incremental', 'generate_enhanced_asg', 'diff_ast', 'find_node_at_position', 'search_code_patterns', 'transform_code_patterns', 'validate_ast_pattern', 'list_transformation_examples', 'sync_file_to_graph', 'query_neo4j_graph', 'ask_uss_agent', 'uss_agent_status', 'analyze_source_file', 'analyze_project']
