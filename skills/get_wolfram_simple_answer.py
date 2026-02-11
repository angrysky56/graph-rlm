"""
WolframAlpha Simple Answer Skill.

Provides a wrapper for the WolframAlpha 'get_simple_answer' tool.
"""

import json
import logging

import requests

logger = logging.getLogger("graph_rlm.skills.get_wolfram_simple_answer")


def get_wolfram_simple_answer(query: str) -> str:
    """
    Retrieves a simple text answer from WolframAlpha for a given query.

    Args:
        query: The natural language question or query for WolframAlpha.

    Returns:
        A JSON string containing the result or an error message.
    """
    server = "wolframalpha"
    tool_name = "get_simple_answer"
    url = f"http://{server}:8000/call"
    data = {"name": tool_name, "arguments": {"query": query}}
    try:
        resp = requests.post(url, json=data, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        return json.dumps(result)
    except requests.exceptions.RequestException as e:
        logger.error("WolframAlpha request failed: %s", e)
        return f"Error: Network or server error occurred: {str(e)}"
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("Failed to parse WolframAlpha response: %s", e)
        return f"Error: Failed to parse response: {str(e)}"
    except Exception as e:  # noqa: BLE001
        logger.error("Unexpected error in get_wolfram_simple_answer: %s", e)
        return f"Error: An unexpected error occurred: {str(e)}"
