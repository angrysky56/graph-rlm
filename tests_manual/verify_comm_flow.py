import asyncio
import contextvars
import queue

from graph_rlm.backend.src.core.agent import Agent, execution_events
from graph_rlm.backend.src.core.sheaf import sheaf


async def verify_comm_flow():
    print("--- Verifying Communication Flow ---")

    # Mock class for Agent to avoid full init
    class MockAgent:
        def __init__(self):
            # Minimal state needed for emit_event
            pass

    # Actually, we need a real Agent instance or a valid mock that has emit_event
    # Let's create a minimal Agent-like object manually to test emit_event logic
    # since we already patched it in agent.py

    # We can use the real Agent class but mock its constructor if needed
    # Or just use a simple object and copy the method for testing

    # Let's use the real Agent but don't call __init__ fully if possible
    # Actually, it's easier to just test via a real Agent if we can

    mock_q = queue.Queue()
    token = execution_events.set(mock_q)

    try:
        # Create a dummy agent (we don't need full init for emit_event)
        agent = Agent.__new__(Agent)

        print("Emitting external event...")
        agent.emit_event("thinking", content="External Thought", is_internal=False)

        print("Emitting internal event...")
        agent.emit_event("thinking", content="Internal Noise", is_internal=True)

        # Check queue
        items = []
        while not mock_q.empty():
            items.append(mock_q.get())

        print(f"Captured {len(items)} events in queue.")
        for i in items:
            print(f" - {i}")

        assert len(items) == 1, "Should only have 1 event in queue (external)"
        assert items[0]["content"] == "External Thought", "Content mismatch"
        print("✅ Communication flow filtering passed!")

    finally:
        execution_events.reset(token)

async def verify_sheaf_robustness():
    print("\n--- Verifying Sheaf Monitor Robustness ---")

    code_with_imports = """
import json
import os
def validate_test(target):
    # Test Path and os
    p = Path("/tmp")
    data = {"test": os.name}
    s = json.dumps(data)
    return True
"""
    print("Testing code with imports and Path...")
    res = await sheaf.check_axiomatic_consistency(code_with_imports, task_tags=["general"])
    print(f"Result: {res['status']}")
    assert res["status"] == "HEALTHY", f"Status should be HEALTHY, got {res['status']}: {res.get('critique')}"

    code_with_await = """
async def test_async():
    res = await rlm.query("test")
    return res == "MOCK_TOOL_OUTPUT"

import asyncio
def validate_async(target):
    # We can't easily wait for async in a sync validator unless we use a bridge
    # But let's see if it handles the call without crashing
    return True
"""
    print("Testing code with async mock check...")
    res = await sheaf.check_axiomatic_consistency(code_with_await, task_tags=["general"])
    print(f"Result: {res['status']}")
    assert res["status"] == "HEALTHY"

    print("✅ Sheaf monitor robustness passed!")

async def main():
    await verify_comm_flow()
    await verify_sheaf_robustness()

if __name__ == "__main__":
    asyncio.run(main())
