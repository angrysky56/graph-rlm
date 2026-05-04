import asyncio
import sys
import uuid
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from unittest.mock import MagicMock

sys.modules["structlog"] = MagicMock()

from graph_rlm.backend.src.core.db import db
from graph_rlm.backend.src.core.scratchpad_builder import scratchpad_builder
from graph_rlm.backend.src.core.state import ExecutionState, agent_state


async def test_context_stability():
    print("\n--- Testing Context Stability (Amnesia Fix) ---")
    session_id = f"test-sess-{uuid.uuid4().hex[:8]}"
    root_session_id = session_id
    round_id_1 = f"{session_id}:Round:1"
    round_id_2 = f"{session_id}:Round:2"

    # 1. Create a thought in Round 1
    db.create_thought_node(
        thought_id=str(uuid.uuid4()),
        prompt="First thought in round 1",
        session_id=session_id,
        root_session_id=root_session_id,
        round_id=round_id_1,
        status="completed",
        step_id=1,
    )

    # 2. Build scratchpad for Round 2 (Simulating a round shift)
    # The scratchpad should now find the round 1 thought via fallback
    pad = await scratchpad_builder.build_scratchpad(
        session_id=session_id,
        root_session_id=root_session_id,
        task="Test Task",
        current_step=2,
        current_round_id=round_id_2,
    )

    print(f"Scratchpad built. Snippet:\n{pad[:300]}...")

    if "First thought in round 1" in pad:
        print("✅ SUCCESS: History found via fallback query.")
    else:
        print("❌ FAILURE: History missing (Amnesia persisted).")


async def test_grounding_verification():
    print("\n--- Testing Grounding Verification (Rule 5 Fix) ---")
    from graph_rlm.backend.src.core.agent import Agent

    agent = Agent()
    state = ExecutionState()
    token = agent_state.set(state)

    try:
        # Simulate a side-effect
        state.pending_side_effects.append("write_to_file")
        print(f"Pending effects: {state.pending_side_effects}")

        # Simulate a Pythonic verification
        code = "if os.path.exists('report.json'): print('Verified')"
        agent._check_verification(code)

        if not state.pending_side_effects:
            print(
                "✅ SUCCESS: Pending effects cleared by Pythonic verification pattern."
            )
        else:
            print(
                f"❌ FAILURE: Pending effects still present: {state.pending_side_effects}"
            )
    finally:
        agent_state.reset(token)


if __name__ == "__main__":
    asyncio.run(test_context_stability())
    asyncio.run(test_grounding_verification())
