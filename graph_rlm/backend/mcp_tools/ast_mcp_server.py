"""
Auto-generated wrapper for ast-mcp-server MCP server.

This module provides Python function wrappers for all tools
exposed by the ast-mcp-server server.

Do not edit manually.
"""

from typing import Any


def parse_to_ast(code: Any | None = None, language: Any | None = None, filename: Any | None = None, **kwargs) -> Any:
    """
Step 1: Parse code → AST (syntax tree).

Use this to validate syntax or get a raw tree dump.
If 'filename' is provided and 'code' is missing or a placeholder, it will read the file.


    Args:
        code: 
        language: 
        filename: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if code is not None:
        mcp_args["code"] = code
    if language is not None:
        mcp_args["language"] = language
    if filename is not None:
        mcp_args["filename"] = filename

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="parse_to_ast",
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


def generate_asg(code: Any | None = None, language: Any | None = None, filename: Any | None = None, **kwargs) -> Any:
    """
Step 3: Parse code → AST → ASG (graph).

Use this to explore basic relationships (edges) between nodes.
If 'filename' is provided and 'code' is missing or a placeholder, it will read the file.


    Args:
        code: 
        language: 
        filename: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if code is not None:
        mcp_args["code"] = code
    if language is not None:
        mcp_args["language"] = language
    if filename is not None:
        mcp_args["filename"] = filename

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="generate_asg",
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


def analyze_code(code: Any | None = None, language: Any | None = None, filename: Any | None = None, **kwargs) -> Any:
    """
Step 2: Extract metadata (Functions, Classes, Imports).

Use this for high-level file summaries.
If 'filename' is provided and 'code' is missing or a placeholder, it will read the file.


    Args:
        code: 
        language: 
        filename: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if code is not None:
        mcp_args["code"] = code
    if language is not None:
        mcp_args["language"] = language
    if filename is not None:
        mcp_args["filename"] = filename

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="analyze_code",
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


def parse_to_ast_incremental(code: Any | None = None, old_code: Any | None = None, language: Any | None = None, filename: Any | None = None, **kwargs) -> Any:
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
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if code is not None:
        mcp_args["code"] = code
    if old_code is not None:
        mcp_args["old_code"] = old_code
    if language is not None:
        mcp_args["language"] = language
    if filename is not None:
        mcp_args["filename"] = filename

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="parse_to_ast_incremental",
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


def generate_enhanced_asg(code: Any | None = None, language: Any | None = None, filename: Any | None = None, **kwargs) -> Any:
    """Step 3 (Enhanced): Deep semantic analysis (Scope, Data Flow). Use for refactoring or complex queries.

    Args:
        code: 
        language: 
        filename: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if code is not None:
        mcp_args["code"] = code
    if language is not None:
        mcp_args["language"] = language
    if filename is not None:
        mcp_args["filename"] = filename

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="generate_enhanced_asg",
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


def diff_ast(old_code: str | Any = None, new_code: str | Any = None, language: Any | None = None, filename: Any | None = None, **kwargs) -> Any:
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
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if old_code is not None:
        mcp_args["old_code"] = old_code
    if new_code is not None:
        mcp_args["new_code"] = new_code
    if language is not None:
        mcp_args["language"] = language
    if filename is not None:
        mcp_args["filename"] = filename

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="diff_ast",
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


def find_node_at_position(code: Any | None = None, line: int | None = None, column: int | None = None, language: Any | None = None, filename: Any | None = None, **kwargs) -> Any:
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
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if code is not None:
        mcp_args["code"] = code
    if line is not None:
        mcp_args["line"] = line
    if column is not None:
        mcp_args["column"] = column
    if language is not None:
        mcp_args["language"] = language
    if filename is not None:
        mcp_args["filename"] = filename

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="find_node_at_position",
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


def search_code_patterns(code: str | Any = None, pattern: str | Any = None, language: Any | None = None, filename: Any | None = None, **kwargs) -> Any:
    """
Search for structural patterns in code using ast-grep.

Returns {matches, count}.
If 'filename' is provided and 'code' is missing or a placeholder, it will read the file.
Use generic patterns like 'async def $FUNC($$$ARGS)' for better discovery.


    Args:
        code: 
        pattern: 
        language: 
        filename: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if code is not None:
        mcp_args["code"] = code
    if pattern is not None:
        mcp_args["pattern"] = pattern
    if language is not None:
        mcp_args["language"] = language
    if filename is not None:
        mcp_args["filename"] = filename

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="search_code_patterns",
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


def transform_code_patterns(code: str | Any = None, pattern: str | Any = None, replacement: str | Any = None, language: Any | None = None, filename: Any | None = None, preview_only: bool | None = None, **kwargs) -> Any:
    """
Replace structural patterns in code using ast-grep.

Returns {transformed_code, changes_applied}.


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
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if code is not None:
        mcp_args["code"] = code
    if pattern is not None:
        mcp_args["pattern"] = pattern
    if replacement is not None:
        mcp_args["replacement"] = replacement
    if language is not None:
        mcp_args["language"] = language
    if filename is not None:
        mcp_args["filename"] = filename
    if preview_only is not None:
        mcp_args["preview_only"] = preview_only

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="transform_code_patterns",
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


def validate_ast_pattern(pattern: str | Any = None, language: str | Any = None, **kwargs) -> Any:
    """Check if ast-grep pattern syntax is valid for the specified language.

    Args:
        pattern: 
        language: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if pattern is not None:
        mcp_args["pattern"] = pattern
    if language is not None:
        mcp_args["language"] = language

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="validate_ast_pattern",
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


def list_transformation_examples(**kwargs) -> Any:
    """Get common ast-grep pattern examples for code modernization and refactoring.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="list_transformation_examples",
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


def sync_file_to_graph(code: str | Any = None, file_path: str | Any = None, language: Any | None = None, project_name: Any | None = None, **kwargs) -> Any:
    """Parse code → store AST+ASG+metrics in Neo4j. Returns {stored: {ast_id, asg_id, analysis_id}}.

    Args:
        code: 
        file_path: 
        language: 
        project_name: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if code is not None:
        mcp_args["code"] = code
    if file_path is not None:
        mcp_args["file_path"] = file_path
    if language is not None:
        mcp_args["language"] = language
    if project_name is not None:
        mcp_args["project_name"] = project_name

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="sync_file_to_graph",
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


def query_neo4j_graph(query: str | Any = None, parameters: Any | None = None, **kwargs) -> Any:
    """Execute Cypher query on code graph. Returns {records, count}.

    Args:
        query: 
        parameters: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if query is not None:
        mcp_args["query"] = query
    if parameters is not None:
        mcp_args["parameters"] = parameters

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="query_neo4j_graph",
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


def ask_uss_agent(query: str | Any = None, **kwargs) -> Any:
    """Graph Query: Ask natural language questions about the codebase (uses Neo4j/ChromaDB).

    Args:
        query: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if query is not None:
        mcp_args["query"] = query

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="ask_uss_agent",
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


def uss_agent_status(**kwargs) -> Any:
    """Check status of the USS Agent services (Neo4j, ChromaDB, LLM).

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="uss_agent_status",
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


def analyze_source_file(project_name: str | Any = None, code: Any | None = None, language: Any | None = None, filename: Any | None = None, include_summary: bool | None = None, output_folder: Any | None = None, **kwargs) -> Any:
    """Analyze a single source file, save reports to disk, and optionally generate an AI summary.

    Args:
        project_name: 
        code: 
        language: 
        filename: 
        include_summary: 
        output_folder: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if project_name is not None:
        mcp_args["project_name"] = project_name
    if code is not None:
        mcp_args["code"] = code
    if language is not None:
        mcp_args["language"] = language
    if filename is not None:
        mcp_args["filename"] = filename
    if include_summary is not None:
        mcp_args["include_summary"] = include_summary
    if output_folder is not None:
        mcp_args["output_folder"] = output_folder

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="analyze_source_file",
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


def analyze_project(project_path: str | Any = None, project_name: str | Any = None, file_extensions: Any | None = None, sync_to_db: bool | None = None, include_summary: bool | None = None, **kwargs) -> Any:
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
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if project_path is not None:
        mcp_args["project_path"] = project_path
    if project_name is not None:
        mcp_args["project_name"] = project_name
    if file_extensions is not None:
        mcp_args["file_extensions"] = file_extensions
    if sync_to_db is not None:
        mcp_args["sync_to_db"] = sync_to_db
    if include_summary is not None:
        mcp_args["include_summary"] = include_summary

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="ast-mcp-server",
            tool_name="analyze_project",
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
    return ['parse_to_ast', 'generate_asg', 'analyze_code', 'parse_to_ast_incremental', 'generate_enhanced_asg', 'diff_ast', 'find_node_at_position', 'search_code_patterns', 'transform_code_patterns', 'validate_ast_pattern', 'list_transformation_examples', 'sync_file_to_graph', 'query_neo4j_graph', 'ask_uss_agent', 'uss_agent_status', 'analyze_source_file', 'analyze_project']
