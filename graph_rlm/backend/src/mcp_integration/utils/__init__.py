"""
Utilities for MCP integration.
"""

from typing import Any


def normalize_mcp_result(result: Any) -> Any:
    """
    Normalizes MCP tool results to make them more agent-friendly.

    If the result is a list containing only one TextContent-like object,
    extracts the text.
    """
    if isinstance(result, list) and len(result) == 1:
        item = result[0]
        # Check for TextContent-like object
        if hasattr(item, "text") and isinstance(item.text, str):
            return item.text
        # Check for raw dict representing TextContent
        if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
            return item["text"]

    return result
