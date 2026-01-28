import requests
from bs4 import BeautifulSoup


def main(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo and return a list of results (Title, Link, Snippet)."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        payload = {"q": query}
        response = requests.post(
            "https://html.duckduckgo.com/html/",
            data=payload,
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        for result in soup.select(".result"):
            if len(results) >= max_results:
                break

            title_elem = result.select_one(".result__a")
            snippet_elem = result.select_one(".result__snippet")

            if title_elem:
                link = title_elem["href"]
                title = title_elem.get_text(strip=True)
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                results.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n---")

        if not results:
            return "No results found."

        return "\n".join(results)

    except Exception as e:
        return f"Error searching {query}: {e}"
