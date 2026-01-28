import requests


def wiki_extract(title: str) -> str:
    url = "https://en.wikipedia.org/w/api.php"
    headers = {
        "User-Agent": "MCP-Coordinator-Agent/1.0 (https://example.com; contact@example.com)"
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
        return f"URL: {fullurl}\\nEXTRACT: {extract[:2000]}..."
    except Exception as e:
        return f"Error: {str(e)}\\nResp preview: {resp.text[:300] if 'resp' in locals() else 'No resp'} "
