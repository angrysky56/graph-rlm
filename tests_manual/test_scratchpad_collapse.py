
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

# Adjust path to find backend modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_rlm.backend.src.core.scratchpad_builder import ScratchpadBuilder

async def test_scratchpad_collapsing():
    print("--- [VERIFICATION] Testing Scratchpad Row Collapsing ---")

    builder = ScratchpadBuilder()
    builder.db = MagicMock()

    # Simulate repetitive thought nodes
    results = [
        {
            "id": f"t{i}",
            "prompt": "list_skills()",
            "status": "success",
            "result": "Skill A, Skill B",
            "created_at": 1700000000000 + (i * 1000),
            "repl_id": "repl-1",
            "turn_id": 1,
            "step_id": i,
        }
        for i in range(1, 6) # 5 identical steps
    ]

    # Mock _generate_step_summary to return identical summary for these
    builder._generate_step_summary = AsyncMock(return_value="Listed skills")

    # Format rows
    table = await builder._format_progress_rows(results)
    print("\nGenerated Table:")
    print(table)

    # Assertions
    assert "**REPETITIVE ACTION**" in table, "Row collapsing failed: Marker missing"
    assert "repeated 5 times" in table, "Row collapsing failed: Count incorrect"
    assert "1.1-5" in table, f"Row collapsing failed: Step range incorrect. Table: {table}"

    print("\n✅ Scratchpad Row Collapsing verified successfully.")

if __name__ == "__main__":
    asyncio.run(test_scratchpad_collapsing())
