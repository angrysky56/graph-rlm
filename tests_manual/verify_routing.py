
import asyncio
import json
import queue
from typing import Any, Optional

from core.agent import Agent, execution_events


async def test_routing():
    agent = Agent()
    q = queue.Queue()
    token = execution_events.set(q)

    print("Testing Event Routing...")

    # 1. Test CODE_RESULT
    agent.emit_event("code_output", content="print('hello')", data={"repl_id": "test_repl"})
    event = q.get()
    print(f"Type: {event['type']}, UI Target: {event.get('ui_target')}, Repl ID: {event.get('repl_id')}")
    assert event['ui_target'] == "CODE_RESULT"

    # 2. Test CHAT_RESPONSE
    agent.emit_event("RLM_FINAL_RESPONSE", content="The answer is 42.")
    event = q.get()
    print(f"Type: {event['type']}, UI Target: {event.get('ui_target')}")
    assert event['ui_target'] == "CHAT_RESPONSE"

    # 3. Test TERMINAL_RAW (Default)
    agent.emit_event("thinking", content="I am thinking...")
    event = q.get()
    print(f"Type: {event['type']}, UI Target: {event.get('ui_target')}")
    assert event['ui_target'] == "TERMINAL_RAW"

    # 4. Test Internal (Should still be TERMINAL_RAW or ignored if logic says so)
    agent.emit_event("thinking", content="secret", is_internal=True)
    event = q.get()
    print(f"Type: {event['type']}, UI Target: {event.get('ui_target')}")
    # Based on logic: ui_target = "TERMINAL_RAW" unless thinking AND not is_internal...
    # Wait, the code said:
    # ui_target = "TERMINAL_RAW"
    # if event_type == "thinking" and not is_internal: ui_target = "TERMINAL_RAW"
    # So it stays TERMINAL_RAW.

    execution_events.reset(token)
    print("Routing Tests Passed!")

if __name__ == "__main__":
    # Add src to path
    import os
    import sys
    sys.path.insert(0, os.path.join(os.getcwd(), "graph_rlm", "backend", "src"))

    # Mocking some dependencies if needed (Agent might need logger/config)
    asyncio.run(test_routing())
