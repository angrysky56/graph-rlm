"""
Auto-generated wrapper for diagram-server MCP server.

This module provides Python function wrappers for all tools
exposed by the diagram-server server.

Do not edit manually.
"""

from typing import Any


def create_diagram(diagram_type: str, content: Any | None = None, name: Any | None = None, use_template: bool | None = None) -> Any:
    """
Create a new Mermaid diagram with AUTOMATIC SAVING.

**KEY IMPROVEMENT**: Diagrams are now automatically saved to disk when created,
preventing data loss from session timeouts or forgetting to save manually.

Args:
    diagram_type: Type of diagram (flowchart, sequence, gantt, class, er, git,
                  pie, journey, mindmap, etc.)
    content: Mermaid syntax content. If not provided, uses template or default
    name: Optional name for the diagram
    use_template: Whether to use a template (default: False)

Returns:
    Diagram ID, content, and auto-save status


    Args:
        diagram_type: 
        content: 
        name: 
        use_template: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict, excluding None values
    mcp_args = {}
    if diagram_type is not None:
        mcp_args["diagram_type"] = diagram_type
    if content is not None:
        mcp_args["content"] = content
    if name is not None:
        mcp_args["name"] = name
    if use_template is not None:
        mcp_args["use_template"] = use_template

    async def _async_call():
        return await call_mcp_tool(
            server_name="diagram-server",
            tool_name="create_diagram",
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


def markdown_to_mindmap(markdown_text: str, name: Any | None = None) -> Any:
    """
Convert markdown text to a mind map diagram with AUTOMATIC SAVING.

Args:
    markdown_text: Markdown formatted text
    name: Optional name for the mind map

Returns:
    Generated mind map in Mermaid syntax with auto-save status


    Args:
        markdown_text: 
        name: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict, excluding None values
    mcp_args = {}
    if markdown_text is not None:
        mcp_args["markdown_text"] = markdown_text
    if name is not None:
        mcp_args["name"] = name

    async def _async_call():
        return await call_mcp_tool(
            server_name="diagram-server",
            tool_name="markdown_to_mindmap",
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


def update_diagram(diagram_id: str, content: str) -> Any:
    """
Update an existing diagram's content with AUTOMATIC SAVING.

Args:
    diagram_id: ID of the diagram to update
    content: New Mermaid syntax content

Returns:
    Updated diagram information with auto-save status


    Args:
        diagram_id: 
        content: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict, excluding None values
    mcp_args = {}
    if diagram_id is not None:
        mcp_args["diagram_id"] = diagram_id
    if content is not None:
        mcp_args["content"] = content

    async def _async_call():
        return await call_mcp_tool(
            server_name="diagram-server",
            tool_name="update_diagram",
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


def save_diagram(diagram_id: str, filepath: Any | None = None) -> Any:
    """
Manually save a diagram to a specific file (in addition to auto-save).

NOTE: Diagrams are now auto-saved when created/updated, but this tool
allows saving to custom locations or re-saving existing diagrams.

Args:
    diagram_id: ID of the diagram to save
    filepath: Optional filepath (defaults to ./diagrams/[id].mmd)

Returns:
    Path where the diagram was saved


    Args:
        diagram_id: 
        filepath: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict, excluding None values
    mcp_args = {}
    if diagram_id is not None:
        mcp_args["diagram_id"] = diagram_id
    if filepath is not None:
        mcp_args["filepath"] = filepath

    async def _async_call():
        return await call_mcp_tool(
            server_name="diagram-server",
            tool_name="save_diagram",
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


def list_diagrams() -> Any:
    """
List all diagrams both in memory and on disk.

Returns:
    List of all diagrams with their metadata


    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict, excluding None values
    mcp_args = {}

    async def _async_call():
        return await call_mcp_tool(
            server_name="diagram-server",
            tool_name="list_diagrams",
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


def get_diagram(diagram_id: str) -> Any:
    """
Get a specific diagram by ID. Loads from disk if not in memory.

Args:
    diagram_id: ID of the diagram to retrieve

Returns:
    Diagram content and metadata


    Args:
        diagram_id: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict, excluding None values
    mcp_args = {}
    if diagram_id is not None:
        mcp_args["diagram_id"] = diagram_id

    async def _async_call():
        return await call_mcp_tool(
            server_name="diagram-server",
            tool_name="get_diagram",
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


def delete_diagram(diagram_id: str) -> Any:
    """
Delete a diagram from memory.

Args:
    diagram_id: ID of the diagram to delete

Returns:
    Deletion confirmation


    Args:
        diagram_id: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict, excluding None values
    mcp_args = {}
    if diagram_id is not None:
        mcp_args["diagram_id"] = diagram_id

    async def _async_call():
        return await call_mcp_tool(
            server_name="diagram-server",
            tool_name="delete_diagram",
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


def list_templates() -> Any:
    """
List all available diagram templates.

Returns:
    List of available templates with descriptions


    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict, excluding None values
    mcp_args = {}

    async def _async_call():
        return await call_mcp_tool(
            server_name="diagram-server",
            tool_name="list_templates",
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


def convert_format_to_diagram(content: str, filename: Any | None = None, target_type: str | None = None, source_format: str | None = None, name: Any | None = None) -> Any:
    """
Convert various file formats to Mermaid diagrams with AUTOMATIC SAVING.

**Multi-Format Support**: JSON→Flowcharts, CSV→Org Charts, Python→Class Diagrams, etc.

Args:
    content: The input content to convert
    filename: Optional filename for format detection
    target_type: Target diagram type (auto, flowchart, mindmap, class, organizational, etc.)
    source_format: Source format (auto, json, csv, python, markdown, plaintext)
    name: Optional name for the generated diagram

Returns:
    Generated diagram with format detection info and auto-save status


    Args:
        content: 
        filename: 
        target_type: 
        source_format: 
        name: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict, excluding None values
    mcp_args = {}
    if content is not None:
        mcp_args["content"] = content
    if filename is not None:
        mcp_args["filename"] = filename
    if target_type is not None:
        mcp_args["target_type"] = target_type
    if source_format is not None:
        mcp_args["source_format"] = source_format
    if name is not None:
        mcp_args["name"] = name

    async def _async_call():
        return await call_mcp_tool(
            server_name="diagram-server",
            tool_name="convert_format_to_diagram",
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


def json_to_flowchart(json_content: str, name: Any | None = None) -> Any:
    """
Convert JSON structure to a flowchart diagram with AUTOMATIC SAVING.

Perfect for visualizing data structures, API responses, or configuration files.

Args:
    json_content: JSON string or content to convert
    name: Optional name for the flowchart

Returns:
    Generated flowchart with auto-save status


    Args:
        json_content: 
        name: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict, excluding None values
    mcp_args = {}
    if json_content is not None:
        mcp_args["json_content"] = json_content
    if name is not None:
        mcp_args["name"] = name

    async def _async_call():
        return await call_mcp_tool(
            server_name="diagram-server",
            tool_name="json_to_flowchart",
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


def csv_to_org_chart(csv_content: str, name: Any | None = None, chart_type: str | None = None) -> Any:
    """
Convert CSV data to organizational or relationship chart with AUTOMATIC SAVING.

Args:
    csv_content: CSV data (expects columns like Name, Role, Department, etc.)
    name: Optional name for the chart
    chart_type: Type of chart (organizational or relationship)

Returns:
    Generated organizational chart with auto-save status


    Args:
        csv_content: 
        name: 
        chart_type: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict, excluding None values
    mcp_args = {}
    if csv_content is not None:
        mcp_args["csv_content"] = csv_content
    if name is not None:
        mcp_args["name"] = name
    if chart_type is not None:
        mcp_args["chart_type"] = chart_type

    async def _async_call():
        return await call_mcp_tool(
            server_name="diagram-server",
            tool_name="csv_to_org_chart",
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


def python_to_class_diagram(python_code: str, name: Any | None = None) -> Any:
    """
Convert Python source code to class diagram with AUTOMATIC SAVING.

Automatically parses classes, methods, and relationships from Python code.

Args:
    python_code: Python source code to analyze
    name: Optional name for the class diagram

Returns:
    Generated class diagram with auto-save status


    Args:
        python_code: 
        name: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict, excluding None values
    mcp_args = {}
    if python_code is not None:
        mcp_args["python_code"] = python_code
    if name is not None:
        mcp_args["name"] = name

    async def _async_call():
        return await call_mcp_tool(
            server_name="diagram-server",
            tool_name="python_to_class_diagram",
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


def detect_file_format(content: str, filename: Any | None = None) -> Any:
    """
Detect the format of input content.

Args:
    content: Content to analyze
    filename: Optional filename for additional context

Returns:
    Detected format and analysis info


    Args:
        content: 
        filename: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict, excluding None values
    mcp_args = {}
    if content is not None:
        mcp_args["content"] = content
    if filename is not None:
        mcp_args["filename"] = filename

    async def _async_call():
        return await call_mcp_tool(
            server_name="diagram-server",
            tool_name="detect_file_format",
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
    return ['create_diagram', 'markdown_to_mindmap', 'update_diagram', 'save_diagram', 'list_diagrams', 'get_diagram', 'delete_diagram', 'list_templates', 'convert_format_to_diagram', 'json_to_flowchart', 'csv_to_org_chart', 'python_to_class_diagram', 'detect_file_format']
