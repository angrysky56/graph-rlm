import json

import requests


def firecrawl_web_search(query: str, limit: int = 5) -> str:
    server = "mcp-server-firecrawl"
    tool_name = "firecrawl_search"
    url = f"http://{server}:8000/call"
    data = {"name": tool_name, "arguments": {"query": query, "limit": limit}}
    try:
        resp = requests.post(url, json=data, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        return json.dumps(result)
    except Exception as e:
        return f"Error: {str(e)}"
