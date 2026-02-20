import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

from graph_rlm.backend.src.core.scratchpad_builder import ScratchpadBuilder

# Configure Logging
logging.basicConfig(level=logging.DEBUG)


async def test_windowing():
    print("\n--- Testing Scratchpad Windowing Logic ---")

    # Mock DB
    mock_db = MagicMock()
    builder = ScratchpadBuilder()
    builder.db = mock_db

    # 1. Create 40 rows of mock data (Exceeding the 25 threshold)
    mock_results = []
    for i in range(1, 41):  # 40 rows
        mock_results.append(
            {
                "id": f"step-{i}",
                "prompt": f"Action {i}",
                "status": "success",
                "result": f"Result {i}",
                "created_at": 1678886400000 + i * 1000,
                "repl_id": "repl_1",
                "turn_id": 1,
                "step_id": i,
                "sheaf_score": 0.0,
                "spectral_energy": 0.0,
                "dreamer_analysis": None,
                "execution_summary": f"Summary {i}",
            }
        )

    # 2. Convert to list-based rows to simulate DB return if needed, but
    # _format_progress_rows handles dicts too if standard fields are present.
    # The code says "if isinstance(row, dict): processed_data.append(row)".
    # So passing dicts is fine.

    print(f"Input Rows: {len(mock_results)}")

    output = await builder._format_progress_rows(mock_results)

    print("\n[Scratchpad Output Snippet]:")
    print(output)

    # 3. Assertions
    # Gap row check
    if "steps hidden" not in output:
        print("❌ FAILED: Gap warning missing")
        return

    if "rlm.recall" not in output:
        print("❌ FAILED: Recall hint missing")
        return

    # Head Check (1-5 should be visible)
    if not ("Summary 1" in output and "Summary 5" in output):
        print("❌ FAILED: Head rows missing")
        return

    # Gap Check (Action 6 should be hidden)
    # The summary for Action 6 is "Summary 6".
    if "Summary 6" in output:
        print("❌ FAILED: Action 6 should be hidden in gap")
        return

    # Tail Check (Last 15 -> 26-40 should be visible)
    # Action 25 might be hidden (40-15=25 start index, so index 25 is item 26).
    # wait: tail is processed_data[-15:].
    # 40 items. Tail is items 26..40.
    # Item 26 has prompt "Action 26" and summary "Summary 26".

    if not ("Summary 26" in output and "Summary 40" in output):
        print("❌ FAILED: Tail rows missing")
        return

    # Truncation Tip Check
    if "History Truncated" not in output:
        print("❌ FAILED: Truncation tip missing")
        return

    print("\n✅ Windowing Logic Verified!")


if __name__ == "__main__":
    asyncio.run(test_windowing())
