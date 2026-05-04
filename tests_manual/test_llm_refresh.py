import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "graph_rlm" / "backend" / "src"))

from core.llm import LLMService


async def test_llm_refresh():
    llm = LLMService()
    print("Initializing client...")
    client = await llm._get_client()
    assert client is not None
    print("Client initialized.")

    print("Refreshing LLM service (calling aclose internally)...")
    await llm.refresh()
    print("Refresh complete.")

    print("Re-initializing client...")
    client2 = await llm._get_client()
    assert client2 is not None
    assert client is not client2
    print("New client initialized successfully.")

    await llm.aclose()
    print("Final cleanup complete.")

if __name__ == "__main__":
    asyncio.run(test_llm_refresh())
