"""
Wikipedia Extraction Skill.

Provides utilities for safely extracting text content and URLs from Wikipedia articles.
"""

import requests


def wiki_extract(title: str) -> str:
    """
    Extracts the introduction and URL of a Wikipedia page given its title.

    Args:
        title: The title of the Wikipedia page.

    Returns:
        A string containing the URL and a summary of the page, or an error message.
    """
    url = "https://en.wikipedia.org/w/api.php"
    headers = {
        "User-Agent": (
            "MCP-Coordinator-Agent/1.0 (https://example.com; contact@example.com)"
        )
    }
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts|info",
        "exintro": True,
        "explaintext": True,
        "inprop": "url",
    }
    resp = None
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code != 200:
            return f"HTTP {resp.status_code}: {resp.text[:200]}"

        data = resp.json()
        if "error" in data:
            return f"Wikipedia error: {data.get('error', {})}"

        pages = data["query"]["pages"]
        page = next(iter(pages.values()))
        if "missing" in page:
            return f"Page not found: {title}"

        extract = page.get("extract", "No extract")
        fullurl = page.get("fullurl", "")
        return f"URL: {fullurl}\nEXTRACT: {extract[:2000]}..."

    except requests.exceptions.RequestException as e:
        preview = (
            f"\nResp preview: {resp.text[:300]}"
            if resp is not None
            else " (No response)"
        )
        return f"Network Error: {str(e)}{preview}"
    except Exception as e:  # noqa: BLE001
        return f"Unexpected Error: {str(e)}"
