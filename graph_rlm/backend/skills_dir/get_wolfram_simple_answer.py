import json

import requests


def get_wolfram_simple_answer(query: str) -> str:
    server = "wolframalpha"
    tool_name = "get_simple_answer"
    url = f"http://{server}:8000/call"
    data = {"name": tool_name, "arguments": {"query": query}}
    try:
        resp = requests.post(url, json=data, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        return json.dumps(result)  # to be safe
    except Exception as e:
        return f"Error: {str(e)}"
