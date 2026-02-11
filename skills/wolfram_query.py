"""
Wolfram Query Skill.

Executes a query using the Wolfram Language MCP server, providing access
to powerful computational and symbolic math capabilities.
"""

import logging
from typing import Any

from graph_rlm.backend.mcp_tools import call_tool

logger = logging.getLogger("graph_rlm.skills.wolfram_query")


async def wolfram_query(query: str, output_format: str = "text") -> str:
    """
    Executes a query using the Wolfram Language MCP server.

    Args:
        query: The Wolfram Language code or natural language query.
        output_format: The desired output format (default: "text").

    Returns:
        The result of the Wolfram computation as a string.
    """
    try:
        # Standardize on call_tool for consistency across the skills library
        result = await call_tool("wolframalpha", "get_simple_answer", {"query": query})
        return str(result)
    except RuntimeError as e:
        logger.error("Wolfram query failed: %s", e)
        return f"Error executing Wolfram query: {e}"
    except Exception as e:  # noqa: BLE001
        logger.error("Unexpected error in wolfram_query: %s", e)
        return f"Error: An unexpected failure occurred: {str(e)}"
