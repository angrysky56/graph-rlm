
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

# Adjust path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_rlm.backend.src.core.scratchpad_builder import ScratchpadBuilder

async def verify_scratchpad_restoration():
    print("--- [VERIFICATION] Testing Scratchpad Restoration ---")

    builder = ScratchpadBuilder()
    builder.db = MagicMock()

    # 1. Mock DB behavior
    builder.db.get_completed_rounds.return_value = []
    builder.db.get_active_thoughts.return_value = []

    # Mock thoughts for _build_current_round_progress
    mock_thought = {
        "id": "node_123",
        "prompt": "ls -la",
        "status": "success",
        "result": "file1.txt",
        "created_at": 1700000000000,
        "repl_id": "iso-1",
        "turn_id": 1,
        "step_id": 1,
        "sheaf_score": 0.1,
        "spectral_energy": 0.1,
        "dreamer_analysis": "Safe.",
        "execution_summary": "Listed files."
    }
    builder.db.query.return_value = [mock_thought]

    print("Building scratchpad...")

    scratchpad = await builder.build_scratchpad(
        session_id="s1",
        root_session_id="r1",
        task="Test task",
        current_step=1,
        max_steps=100,
        current_round_id="round_1",
        morph_gestalt="EXISTENCE: [file1.txt]"
    )

    print("\n--- Output Preview ---")
    print(scratchpad)
    print("--- End Preview ---")

    # Assertions
    assert "## Execution Trace (Current Round)" in scratchpad, "Execution Trace section missing!"
    assert "| Time | REPL | T.S | St | Summary" in scratchpad, "Trace table missing!"
    assert "recall('node_123')" in scratchpad, "Recall ID missing from table!"
    assert "## Thimac Gestalt" in scratchpad, "Gestalt section missing!"

    print("\n✅ Scratchpad restoration verified successfully.")

if __name__ == "__main__":
    asyncio.run(verify_scratchpad_restoration())
