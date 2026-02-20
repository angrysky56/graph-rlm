import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure we can import graph_rlm
sys.path.insert(0, os.path.join(os.getcwd(), "graph_rlm/backend"))

from graph_rlm.backend.src.core.llm import LLMService


async def test_malformed_function_call():
    print("--- Testing MALFORMED_FUNCTION_CALL Resilience ---")
    llm = LLMService()

    # Mock data with finish_reason: error and native_finish_reason: MALFORMED_FUNCTION_CALL
    mock_response_data = {
        "id": "gen-test",
        "choices": [
            {
                "finish_reason": "error",
                "native_finish_reason": "MALFORMED_FUNCTION_CALL",
                "message": {"content": ""},
            }
        ],
        "usage": {"total_tokens": 0},
    }

    # Create a real-looking mock response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json = MagicMock(return_value=mock_response_data)
    mock_resp.raise_for_status = MagicMock()
    mock_resp.text = json.dumps(mock_response_data)

    # Mock the AsyncClient.post to return this response
    # Since it's an async method, it must return a coroutine that resolves to mock_resp
    async def mock_post(*args, **kwargs):
        return mock_resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        res = await llm._generate_async([{"role": "user", "content": "test prompt"}])
        print(f"Result: '{res}'")

        if "MALFORMED_FUNCTION_CALL" in res or "Python code blocks" in res:
            print(
                "SUCCESS: LLMService caught the error and returned a corrective action."
            )
        else:
            print("FAILURE: LLMService did not return the expected corrective action.")


if __name__ == "__main__":
    asyncio.run(test_malformed_function_call())
