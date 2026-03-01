import asyncio
import os
import sys
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from graph_rlm.backend.src.core.scratchpad_builder import ScratchpadBuilder
from graph_rlm.backend.src.core.thimac_memory import (
    ThimacEvent,
    ThimacLevel,
    ThimacOperation,
)


async def test_thimac_event_handling():
    print("Testing ThimacEvent handling in ScratchpadBuilder...")
    builder = ScratchpadBuilder()

    # Mock DB
    builder.db = MagicMock()
    builder.db.get_completed_rounds.return_value = []
    builder.db.query.return_value = []

    # Create a dummy ThimacEvent
    event = ThimacEvent(
        thought_id="test-id",
        operation=ThimacOperation.PROCESS,
        level=ThimacLevel.SUBSISTENCE,
        status="success",
        full_data="test prompt",
        result="test result",
        timestamp=1000,
        session_id="test-sid",
        root_session_id="test-rsid",
        round_id="test-rid",
    )

    # Simulate building scratchpad with the event in trajectory
    try:
        await builder.build_scratchpad(
            session_id="test-sid",
            root_session_id="test-rsid",
            task="test task",
            current_round_id="test-rid",
            memory_trajectory=[event],
        )
        print("Success: Scratchpad built without error.")
        # print(f"Pad output snippet: {pad[:200]}...")
    except AttributeError as e:
        print(f"FAILED: Found AttributeError: {e}")
        sys.exit(1)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(test_thimac_event_handling())
