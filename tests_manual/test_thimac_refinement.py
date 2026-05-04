import asyncio
import sys
import uuid
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from graph_rlm.backend.src.core.thimac_memory import (
    ThimacIntention,
    ThimacLevel,
    ThimacMemory,
    ThimacOperation,
)


async def test_thimac_classification():
    print("Testing Thimac Classification Refinements...")
    memory = ThimacMemory()

    # Simulating a read action (ARRIVE)
    thought_read = {
        "id": str(uuid.uuid4()),
        "prompt": "ls -la /home/ty",
        "status": "success",
        "result": "total 0\ndrwxr-xr-x 2 ty ty 40 Feb 23 23:40 .",
        "turn_id": 1,
        "step_id": 1,
    }
    tool_calls = ["view_file"]

    event = memory.ingest_thought(thought_read, tool_calls=tool_calls)

    print("Event 1 (Read):")
    print(f"  Operation: {event.operation} (Expected: ARRIVE)")
    print(f"  Reason: {event.operation_reason}")
    print(f"  Level: {event.level} (Expected: EXISTENCE)")
    print(f"  Intent: {event.intent_type} (Expected: MOTOR)")
    assert event.operation == ThimacOperation.ARRIVE
    assert event.level == ThimacLevel.EXISTENCE
    assert event.intent_type == ThimacIntention.MOTOR

    # Simulating a thought (PROCESS/SUBSISTENCE)
    thought_think = {
        "id": str(uuid.uuid4()),
        "prompt": "I need to analyze the files I found.",
        "status": "pending",
        "turn_id": 1,
        "step_id": 2,
    }

    event2 = memory.ingest_thought(thought_think)
    print("\nEvent 2 (Think):")
    print(f"  Operation: {event2.operation} (Expected: PROCESS)")
    print(f"  Reason: {event2.operation_reason}")
    print(f"  Level: {event2.level} (Expected: SUBSISTENCE)")
    print(f"  Intent: {event2.intent_type} (Expected: PROXIMAL)")
    assert event2.operation == ThimacOperation.PROCESS
    assert event2.level == ThimacLevel.SUBSISTENCE
    assert event2.intent_type == ThimacIntention.PROXIMAL

    # Check Gestalt visibility
    gestalt = memory.get_gestalt_string()
    print(f"\nGestalt Summary:\n{gestalt}")
    assert "Triggered by ingestion tool" in gestalt
    assert "Intent: MOTOR" in gestalt

    print("\nThimac Refinement Tests Passed!")


if __name__ == "__main__":
    asyncio.run(test_thimac_classification())
