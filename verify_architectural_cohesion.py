import asyncio
import sys
import uuid
from unittest.mock import MagicMock, patch

# Mock dependencies to allow importing modules
mock_db = MagicMock()
sys.modules["graph_rlm.backend.src.core.llm"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.dream"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.sheaf"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.omcd"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.navigator"] = MagicMock()
sys.modules["graph_rlm.backend.src.mcp_integration.runtime"] = MagicMock()
sys.modules["graph_rlm.backend.src.mcp_integration.skill_storage"] = MagicMock()

# Import the actual classes/objects we want to test
from graph_rlm.backend.src.core.db import GraphClient
from graph_rlm.backend.src.core.meta_agents import (
    CollaborationState,
    Fragment,
    meta_agents,
)
from graph_rlm.backend.src.core.scratchpad_builder import scratchpad_builder
from graph_rlm.backend.src.core.thimac_memory import (
    ThimacEvent,
    ThimacLevel,
    ThimacMemory,
    ThimacOperation,
)


async def verify():
    print("🚀 Starting Architectural Cohesion Verification...")

    # 1. Verify Meta-Agent Context Propagation
    print("\n--- 1. Meta-Agent Context Propagation ---")
    mock_db_instance = MagicMock()
    meta_agents.db = mock_db_instance

    root_id = "test-root-123"
    round_id = "round-abc"
    turn_id = 5

    state = meta_agents.start_collaboration(
        root_session_id=root_id,
        task="Test meta task",
        round_id=round_id,
        turn_id=turn_id,
    )

    print(f"CollaborationState Round ID: {state.round_id}")
    print(f"CollaborationState Turn ID: {state.turn_id}")

    assert state.round_id == round_id
    assert state.turn_id == turn_id

    # Verify create_thought_node was called with correct context
    # It might be in kwargs
    found = False
    for call in mock_db_instance.create_thought_node.call_args_list:
        if call.kwargs.get("round_id") == round_id:
            found = True
            break
    assert found
    print("✅ Meta-Agent start_collaboration Context OK.")

    # 2. Verify Fragment Context Propagation
    print("\n--- 2. Fragment Context Propagation ---")
    fragment = Fragment(
        session_id="child-sid",
        summary="Test fragment",
        subtopics=[],
        confidence=0.8,
        raw_output="Raw data",
    )

    meta_agents.register_fragment(root_id, fragment)
    # Check if DB call for fragment creation used the state's round_id
    found = False
    for call in mock_db_instance.create_thought_node.call_args_list:
        if (
            call.kwargs.get("status") == "fragment"
            and call.kwargs.get("round_id") == round_id
        ):
            found = True
            break
    assert found
    print("✅ Fragment Context Propagation OK.")

    # 3. Verify Memory Synchronization
    print("\n--- 3. Memory Synchronization ---")
    parent_memory = ThimacMemory()
    child_memory = ThimacMemory()

    event1 = ThimacEvent(
        thought_id="t1",
        operation=ThimacOperation.PROCESS,
        level=ThimacLevel.EXISTENCE,
        status="success",
        utility_score=0.9,
    )
    parent_memory.store(event1)

    # Extract high utility for child
    snapshot = parent_memory.get_high_utility_events()
    print(f"Parent Snapshot Size: {len(snapshot)}")
    assert len(snapshot) == 1

    # Merge into child
    child_memory.merge_events(snapshot)
    print(f"Child Memory Size: {len(child_memory.all_events)}")
    assert len(child_memory.all_events) == 1
    assert child_memory.all_events[0].thought_id == "t1"

    # Simulate child unique event
    event2 = ThimacEvent(
        thought_id="t2",
        operation=ThimacOperation.RELEASE,
        level=ThimacLevel.SUBSISTENCE,
        status="success",
        compression_gain=0.6,  # High utility via compression
    )
    child_memory.store(event2)

    # Back propagate
    child_critical = child_memory.get_high_utility_events(threshold=0.7)
    print(f"Child Critical Size: {len(child_critical)}")
    assert len(child_critical) == 2  # t1 (via merge) and t2 (newly created)

    parent_memory.merge_events(child_critical)
    print(f"Parent Final Memory Size: {len(parent_memory.all_events)}")
    assert len(parent_memory.all_events) == 2
    assert "t2" in [e.thought_id for e in parent_memory.all_events]
    print("✅ Memory Synchronization OK.")

    print("\n🎉 All Architectural Cohesion Checks Passed!")


if __name__ == "__main__":
    asyncio.run(verify())
