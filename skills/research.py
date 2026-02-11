"""
Research Skill.

Provides utilities for searching Arxiv and other academic sources to gather
structured data on specific scientific or technical topics.
"""

import json
import logging
from typing import Any, Dict, List

from graph_rlm.backend.mcp_tools import call_tool

logger = logging.getLogger("graph_rlm.skills.research")


async def research_topic(topic: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search Arxiv for papers on a topic and return structured data.

    Args:
        topic: The research topic to search for.
        max_results: Maximum number of papers to return.

    Returns:
        List of dictionaries containing paper details (title, id, summary, etc.).
    """
    try:
        # Call the MCP tool using the standardized wrapper
        results = await call_tool(
            "arxiv_mcp_server",
            "search_papers",
            {"query": topic, "max_results": max_results},
        )

        # 1. Handle MCP Content objects (TextContent)
        # Often the server returns a list of TextContent objects.
        if (
            isinstance(results, list)
            and len(results) > 0
            and hasattr(results[0], "text")
        ):
            full_text = "".join([r.text for r in results if hasattr(r, "text")])
            try:
                parsed = json.loads(full_text)
                return [parsed] if isinstance(parsed, dict) else parsed
            except json.JSONDecodeError:
                logger.error(
                    "Failed to parse Arxiv response as JSON: %s...", full_text[:200]
                )
                # Return raw text wrapped in a dict if parsing fails
                return [{"title": "Raw Results", "summary": full_text}]

        # 2. Handle direct list/dict return
        if isinstance(results, list):
            # If it's a list of strings, it might be an error or unexpected format
            if len(results) > 0 and isinstance(results[0], str):
                logger.warning(
                    "Received list of strings, likely error or raw text: %s",
                    results[0][:100],
                )
                return [{"title": "Raw Text Result", "content": "\n".join(results)}]
            return results

        if isinstance(results, dict):
            return [results]

        return []

    except RuntimeError as e:
        logger.error("Research failed for topic '%s': %s", topic, e)
        return []
    except Exception as e:  # noqa: BLE001
        logger.error("Unexpected error researching topic '%s': %s", topic, e)
        return []
