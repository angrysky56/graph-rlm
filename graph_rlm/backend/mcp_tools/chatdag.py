"""
Auto-generated wrapper for chatdag MCP server.

This module provides Python function wrappers for all tools
exposed by the chatdag server.

Do not edit manually.
"""

from typing import Any


def search_knowledge(query: str, k: Any | None = None) -> Any:
    """Search the holographic knowledge base for relevant context.
Returns formatted string of results.

    Args:
        query: 
        k: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if query is not None:
        params["query"] = query
    if k is not None:
        params["k"] = k


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="chatdag",
            tool_name="search_knowledge",
            arguments=params,
        )
    return asyncio.run(_async_call())


def crystallize_thought(topic: str) -> Any:
    """Perform deep reasoning (Su Hui Crystallization) on a topic using the knowledge base.
Returns synthesized insight.

    Args:
        topic: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if topic is not None:
        params["topic"] = topic


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="chatdag",
            tool_name="crystallize_thought",
            arguments=params,
        )
    return asyncio.run(_async_call())


def ingest_data(target_path: Any | None = None, recursive: bool | None = None, analyze_images: Any | None = None) -> Any:
    """Ingest files or directories into the knowledge base.
Set recursive=True (default) to scan all subdirectories.
Works with code, docs, PDFs (OCR enabled), and images.

Args:
    path: Path to file or directory to ingest (also accepts target_path)

    Args:
        target_path: 
        recursive: 
        analyze_images: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if target_path is not None:
        params["target_path"] = target_path
    if recursive is not None:
        params["recursive"] = recursive
    if analyze_images is not None:
        params["analyze_images"] = analyze_images


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="chatdag",
            tool_name="ingest_data",
            arguments=params,
        )
    return asyncio.run(_async_call())


def analyze_image(image_path: str, prompt: Any | None = None, detail: str | None = None) -> Any:
    """Analyze an image using the Vision-Language model.

Use this for:
- Understanding charts, graphs, diagrams
- OCR/text extraction from images
- Describing visual content
- Answering questions about images

Args:
    image_path: Absolute path to image file (png, jpg, jpeg, webp, bmp)
    prompt: Optional specific question about the image. Default: general description
    detail: "low" (faster/cheaper) or "high" (more accurate)

Returns:
    VL model's analysis/response

    Args:
        image_path: 
        prompt: 
        detail: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if image_path is not None:
        params["image_path"] = image_path
    if prompt is not None:
        params["prompt"] = prompt
    if detail is not None:
        params["detail"] = detail


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="chatdag",
            tool_name="analyze_image",
            arguments=params,
        )
    return asyncio.run(_async_call())


def extract_pdf_text(pdf_path: str, analyze_images: Any | None = None) -> Any:
    """Extract text from a PDF file.

Uses:
- Embedded text extraction (fast)
- OCR for scanned pages
- Optional VL model for figures/charts

Args:
    pdf_path: Absolute path to PDF file
    analyze_images: If True, use VL model to analyze figures (slower but thorough)

Returns:
    Extracted text content

    Args:
        pdf_path: 
        analyze_images: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if pdf_path is not None:
        params["pdf_path"] = pdf_path
    if analyze_images is not None:
        params["analyze_images"] = analyze_images


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="chatdag",
            tool_name="extract_pdf_text",
            arguments=params,
        )
    return asyncio.run(_async_call())


def feed_data(content: str, source_id: str, metadata: Any | None = None) -> Any:
    """Directly ingest text content into the knowledge base.

Perfect for:
- Conversation context (current discussion)
- API responses (web search, tool outputs)
- Ephemeral insights (thoughts without files)

Args:
    content: Text to ingest
    source_id: Unique identifier (e.g., "conversation/date/topic")
    metadata: Optional context (type, tags, etc.) - currently logged only.

Returns:
    Confirmation with voxel count

    Args:
        content: 
        source_id: 
        metadata: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if content is not None:
        params["content"] = content
    if source_id is not None:
        params["source_id"] = source_id
    if metadata is not None:
        params["metadata"] = metadata


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="chatdag",
            tool_name="feed_data",
            arguments=params,
        )
    return asyncio.run(_async_call())


def clean_corrupted_data(dry_run: bool | None = None, corruption_threshold: Any | None = None, remove_duplicates: Any | None = None, show_diff: Any | None = None) -> Any:
    """Clean corrupted voxels (bad OCR) and remove duplicate sources.

Args:
    dry_run: Report only, do not delete.
    corruption_threshold: Min (cid: occurrences to flag corruption.
    remove_duplicates: Detect and remove older duplicate files.
    show_diff: Show content diff for duplicates.

Returns:
    Summary of actions taken.

    Args:
        dry_run: 
        corruption_threshold: 
        remove_duplicates: 
        show_diff: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if dry_run is not None:
        params["dry_run"] = dry_run
    if corruption_threshold is not None:
        params["corruption_threshold"] = corruption_threshold
    if remove_duplicates is not None:
        params["remove_duplicates"] = remove_duplicates
    if show_diff is not None:
        params["show_diff"] = show_diff


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="chatdag",
            tool_name="clean_corrupted_data",
            arguments=params,
        )
    return asyncio.run(_async_call())


def diff_sources(source_a: str, source_b: str) -> Any:
    """Compare content between two source URIs.
Useful for checking version changes.

Args:
    source_a: First source URI (partial match allowed).
    source_b: Second source URI (partial match allowed).

Returns:
    Unified diff summary.

    Args:
        source_a: 
        source_b: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if source_a is not None:
        params["source_a"] = source_a
    if source_b is not None:
        params["source_b"] = source_b


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="chatdag",
            tool_name="diff_sources",
            arguments=params,
        )
    return asyncio.run(_async_call())


def train_adaptive_model(force: bool | None = None, training_threshold: Any | None = None) -> Any:
    """Train the adaptive neural network to optimize retrieval.
Use this when retrieval quality is poor.

Args:
    force: Ignore minimum data requirements.
    training_threshold: Resonance score threshold for "good" outcomes (default: 1.5).

Returns:
    Training status.

    Args:
        force: 
        training_threshold: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if force is not None:
        params["force"] = force
    if training_threshold is not None:
        params["training_threshold"] = training_threshold


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="chatdag",
            tool_name="train_adaptive_model",
            arguments=params,
        )
    return asyncio.run(_async_call())


def report_search_quality(query: str, was_useful: bool, comments: Any | None = None) -> Any:
    """Feedback loop: Rate a recent search to improve future performance.

Args:
    query: The query text to rate (approximate match).
    was_useful: Boolean rating (True=Good, False=Bad).
    comments: Optional context.

Returns:
    Confirmation status.

    Args:
        query: 
        was_useful: 
        comments: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if query is not None:
        params["query"] = query
    if was_useful is not None:
        params["was_useful"] = was_useful
    if comments is not None:
        params["comments"] = comments


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="chatdag",
            tool_name="report_search_quality",
            arguments=params,
        )
    return asyncio.run(_async_call())



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['search_knowledge', 'crystallize_thought', 'ingest_data', 'analyze_image', 'extract_pdf_text', 'feed_data', 'clean_corrupted_data', 'diff_sources', 'train_adaptive_model', 'report_search_quality']
