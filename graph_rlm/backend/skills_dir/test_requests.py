import requests


def test_requests() -> str:
    try:
        resp = requests.get("https://www.google.com", timeout=5)
        return f"Status: {resp.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"
