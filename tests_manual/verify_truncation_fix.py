
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

# Adjust path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_rlm.backend.src.core.agent import Agent

async def test_truncation_summary():
    print("--- [VERIFICATION] Testing Thimac Truncation Summary ---")

    agent = Agent()
    agent.db = MagicMock()
    agent.llm = AsyncMock()
    agent.runtime = AsyncMock()

    # Mock clean execution environment
    agent.active_repls = {}

    # 1. Simulate LARGE Output
    large_output = "X" * 5000
    agent.runtime.execute.return_value = (large_output, "", "Result", 0)

    print("Executing code with 5000 chars of output...")

    output, failed, summary = await agent._execute_code(
        code="print('large')",
        thought_id="t1",
        session_id="s1"
    )

    print(f"\nUser Output Length: {len(output)}")
    print(f"User Output Start: {output[:50]}...")

    print(f"\nExecution Summary: {summary}")

    # Assertions
    assert output.startswith("[Output truncated due to size"), "User output not truncated correctly"
    assert summary is not None, "Execution summary is None!"
    assert summary.startswith("[Truncated Output]: XXXX"), "Summary content incorrect"
    assert len(summary) < 500, "Summary too long"

    print("\n✅ Truncation logic verified successfully.")

if __name__ == "__main__":
    asyncio.run(test_truncation_summary())
