import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph_rlm.backend.src.core.agent import Agent


async def verify_recovery():
    print("--- Verifying Agent Recovery ---")
    agent = Agent()

    # 1. Verify Stop Signal Logic
    print("Testing stop signal...")
    agent.stop_generation()
    assert agent.global_stop_event.is_set()
    assert agent.stop_requested
    print("✓ Stop signal set correctly.")

    # 2. Verify Stop signal clearing on new query
    print("Testing stop signal clearing...")
    # Mocking dependencies for query_sync
    agent.db = MagicMock()
    agent.llm = MagicMock()
    agent.llm.generate = AsyncMock(return_value="RLM_FINAL_RESPONSE: Test")

    # We don't actually run the loop, just verify the start of it
    # We can use a small max_steps or mock the whole thing
    try:
        await asyncio.wait_for(agent.query_sync(prompt="test"), timeout=1.0)
    except Exception as e:
        print(f"Query sync expectedly exited or timed out: {e}")

    assert not agent.global_stop_event.is_set()
    print("✓ Stop signal cleared on query start.")

    # 3. Verify Error Visibility
    print("Testing error visibility...")
    agent.llm.generate = AsyncMock(side_effect=Exception("Simulated LLM Error"))

    events = []
    agent.emit_event = MagicMock(
        side_effect=lambda t, content=None, **kwargs: events.append((t, content))
    )

    try:
        # Run one step
        await agent.query_sync(prompt="test")
    except Exception:
        pass

    error_events = [e for e in events if e[0] == "error"]
    assert len(error_events) > 0
    assert "Simulated LLM Error" in error_events[0][1]
    print("✓ Error event emitted correctly.")

    print("\n--- All Tests Passed! ---")


if __name__ == "__main__":
    asyncio.run(verify_recovery())
