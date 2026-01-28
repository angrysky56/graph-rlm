"""
Auto-generated wrapper for arxiv-mcp-server MCP server.

This module provides Python function wrappers for all tools
exposed by the arxiv-mcp-server server.

Do not edit manually.
"""

from typing import Any


def search_papers(query: str, max_results: int | None = None, date_from: str | None = None, date_to: str | None = None, categories: list[str] | None = None, sort_by: str | None = None) -> Any:
    """Search for papers on arXiv with advanced filtering and query optimization.

QUERY CONSTRUCTION GUIDELINES:
- Use QUOTED PHRASES for exact matches: "multi-agent systems", "neural networks", "machine learning"
- Combine related concepts with OR: "AI agents" OR "software agents" OR "intelligent agents"  
- Use field-specific searches for precision:
  - ti:"exact title phrase" - search in titles only
  - au:"author name" - search by author
  - abs:"keyword" - search in abstracts only
- Use ANDNOT to exclude unwanted results: "machine learning" ANDNOT "survey"
- For best results, use 2-4 core concepts rather than long keyword lists

ADVANCED SEARCH PATTERNS:
- Field + phrase: ti:"transformer architecture" for papers with exact title phrase
- Multiple fields: au:"Smith" AND ti:"quantum" for author Smith's quantum papers  
- Exclusions: "deep learning" ANDNOT ("survey" OR "review") to exclude survey papers
- Broad + narrow: "artificial intelligence" AND (robotics OR "computer vision")

CATEGORY FILTERING (highly recommended for relevance):
- cs.AI: Artificial Intelligence
- cs.MA: Multi-Agent Systems  
- cs.LG: Machine Learning
- cs.CL: Computation and Language (NLP)
- cs.CV: Computer Vision
- cs.RO: Robotics
- cs.HC: Human-Computer Interaction
- cs.CR: Cryptography and Security
- cs.DB: Databases

EXAMPLES OF EFFECTIVE QUERIES:
- ti:"reinforcement learning" with categories: ["cs.LG", "cs.AI"] - for RL papers by title
- au:"Hinton" AND "deep learning" with categories: ["cs.LG"] - for Hinton's deep learning work
- "multi-agent" ANDNOT "survey" with categories: ["cs.MA"] - exclude survey papers
- abs:"transformer" AND ti:"attention" with categories: ["cs.CL"] - attention papers with transformer abstracts

DATE FILTERING: Use YYYY-MM-DD format for historical research:
- date_to: "2015-12-31" - for foundational/classic work (pre-2016)
- date_from: "2020-01-01" - for recent developments (post-2020)
- Both together for specific time periods

RESULT QUALITY: Results sorted by RELEVANCE (most relevant papers first), not just newest papers.
This ensures you get the most pertinent results regardless of publication date.

TIPS FOR FOUNDATIONAL RESEARCH:
- Use date_to: "2010-12-31" to find classic papers on BDI, SOAR, ACT-R
- Combine with field searches: ti:"BDI" AND abs:"belief desire intention"  
- Try author searches: au:"Rao" AND "BDI" for Anand Rao's foundational BDI work

    Args:
        query: Search query using quoted phrases for exact matches (e.g., '"machine learning" OR "deep learning"') or specific technical terms. Avoid overly broad or generic terms.
        max_results: Maximum number of results to return (default: 10, max: 50). Use 15-20 for comprehensive searches.
        date_from: Start date for papers (YYYY-MM-DD format). Use to find recent work, e.g., '2023-01-01' for last 2 years.
        date_to: End date for papers (YYYY-MM-DD format). Use with date_from to find historical work, e.g., '2020-12-31' for older research.
        categories: Strongly recommended: arXiv categories to focus search (e.g., ['cs.AI', 'cs.MA'] for agent research, ['cs.LG'] for ML, ['cs.CL'] for NLP, ['cs.CV'] for vision). Greatly improves relevance.
        sort_by: Sort results by 'relevance' (most relevant first, default) or 'date' (newest first). Use 'relevance' for focused searches, 'date' for recent developments.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if query is not None:
        params["query"] = query
    if max_results is not None:
        params["max_results"] = max_results
    if date_from is not None:
        params["date_from"] = date_from
    if date_to is not None:
        params["date_to"] = date_to
    if categories is not None:
        params["categories"] = categories
    if sort_by is not None:
        params["sort_by"] = sort_by


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="arxiv-mcp-server",
            tool_name="search_papers",
            arguments=params,
        )
    return asyncio.run(_async_call())


def download_paper(paper_id: str, check_status: bool | None = None) -> Any:
    """Download a paper and create a resource for it

    Args:
        paper_id: The arXiv ID of the paper to download
        check_status: If true, only check conversion status without downloading

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if paper_id is not None:
        params["paper_id"] = paper_id
    if check_status is not None:
        params["check_status"] = check_status


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="arxiv-mcp-server",
            tool_name="download_paper",
            arguments=params,
        )
    return asyncio.run(_async_call())


def list_papers() -> Any:
    """List all existing papers available as resources

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="arxiv-mcp-server",
            tool_name="list_papers",
            arguments=params,
        )
    return asyncio.run(_async_call())


def read_paper(paper_id: str) -> Any:
    """Read the full content of a stored paper in markdown format

    Args:
        paper_id: The arXiv ID of the paper to read

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if paper_id is not None:
        params["paper_id"] = paper_id


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="arxiv-mcp-server",
            tool_name="read_paper",
            arguments=params,
        )
    return asyncio.run(_async_call())



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['search_papers', 'download_paper', 'list_papers', 'read_paper']
