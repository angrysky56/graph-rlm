import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add project root to sys.path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from graph_rlm.backend.src.core.agent import Agent


async def test_thimac_restoration():
    print("Testing Thimac Restoration after Reflexion...")
    agent = Agent()

    # 1. Simulate a thought
    thought_id = "t1"
    prompt = "print('Hello')"
    status = "success"
    result = "Hello"
    step = 1
    repl_id = "REPL1"

    agent._sync_thimac(
        thought_id=thought_id,
        prompt=prompt,
        status=status,
        result=result,
        step=step,
        repl_id=repl_id
    )

    gestalt = agent.morph_memory.get_gestalt_string()
    print("\nGestalt after Step 1:")
    print(gestalt)
    assert "PROCESS: Hello" in gestalt or "CREATE" in gestalt or "RECEIVE" in gestalt or "PROCESS" in gestalt

    # 2. Simulate a Dreamer Rejection (Reflexion)
    feedback_id = "t2"
    feedback_prompt = "DREAMER REJECTION: Be more specific."

    agent._sync_thimac(
        thought_id=feedback_id,
        prompt=feedback_prompt,
        status="reflexion",
        result=None,
        step=step, # Reflexion usually happens on the same step
        repl_id=repl_id
    )

    gestalt_after = agent.morph_memory.get_gestalt_string()
    print("\nGestalt after Dreamer Rejection (Reflexion):")
    print(gestalt_after)

    # Verify that the previous state (Step 1) is still present
    assert "active results" in gestalt_after
    assert "PROCESS" in gestalt_after

    print("\n✅ Thimac restoration verified.")

if __name__ == "__main__":
    asyncio.run(test_thimac_restoration())
