
import os
import sys
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "graph_rlm/backend/src"))

from core.scratchpad_builder import ScratchpadBuilder


def test_truncation():
    builder = ScratchpadBuilder()
    # Mock the DB
    builder.db = MagicMock()

    # Generate 50 mock results
    mock_results = []
    for i in range(1, 51):
        mock_results.append({
            "id": f"node_{i}",
            "prompt": f"prompt {i}",
            "status": "success",
            "result": f"result {i}",
            "created_at": 1700000000000 + i * 1000,
            "repl_id": "repl_1",
            "execution_summary": f"summary {i}",
            "next_action": None,
            "dreamer_analysis": None,
            "final_response": None,
            "turn_id": 1,
            "step_id": i,
            "code_hash": "hash"
        })

    # Test formatting
    output = builder._format_progress_rows(mock_results, max_rows=30)

    print("--- Scratchpad Output (Truncated) ---")
    print(output)

    # Check if header steps are there
    assert "Step 1" in output
    assert "Step 5" in output
    # Check if tail steps are there
    assert "Step 31" in output
    assert "Step 50" in output
    # Check if omitted line is there
    assert "steps omitted" in output
    # Check if middle steps are NOT there
    assert "Step 6" not in output
    assert "Step 30" not in output # 50 - 20 = 30

    print("\nTruncation verification successful!")

def test_no_truncation():
    builder = ScratchpadBuilder()
    builder.db = MagicMock()

    mock_results = []
    for i in range(1, 11):
        mock_results.append({
            "id": f"node_{i}",
            "prompt": f"prompt {i}",
            "status": "success",
            "result": f"result {i}",
            "created_at": 1700000000000 + i * 1000,
            "repl_id": "repl_1",
            "execution_summary": f"summary {i}",
            "turn_id": 1,
            "step_id": i,
        })

    output = builder._format_progress_rows(mock_results, max_rows=30)
    print("\n--- Scratchpad Output (No Truncation) ---")
    print(output)

    assert "Step 1" in output
    assert "Step 10" in output
    assert "steps omitted" not in output
    print("\nNo Truncation verification successful!")

if __name__ == "__main__":
    try:
        test_truncation()
        test_no_truncation()
    except AssertionError as e:
        print(f"Verification FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
