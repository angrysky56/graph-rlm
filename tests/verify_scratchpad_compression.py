import datetime
from unittest.mock import MagicMock

from graph_rlm.backend.src.core.scratchpad_builder import ScratchpadBuilder


def test_condensation():
    sb = ScratchpadBuilder()
    sb.db = MagicMock()

    # 1. Mock completed rounds
    sb.db.get_completed_rounds.return_value = []

    # 2. Mock results with repetitive rejections
    # status: b[2], prompt: b[1], dreamer_analysis: b[8]
    mock_results = [
        ["id1", "Thought 1", "success", "Result 1", 1000, "repl1", "summary1", "next1", "", ""],
        ["id2", "Hallucinated Thought", "rejected", "Rejected Result", 2000, "repl1", "summary2", "next2", "Mirror Hallucination detected", ""],
        ["id3", "Hallucinated Thought", "rejected", "Rejected Result", 3000, "repl1", "summary3", "next3", "Mirror Hallucination detected", ""],
        ["id4", "Hallucinated Thought", "rejected", "Rejected Result", 4000, "repl1", "summary4", "next4", "Mirror Hallucination detected", ""],
        ["id5", "Final Correct Thought", "success", "Final Result", 5000, "repl1", "summary5", "next5", "", "Final Output"]
    ]

    sb.db.query.return_value = [] # for audits and neighbors

    # Inject results into the builder's logic
    output = sb._format_progress_rows(mock_results)

    print("--- VERIFICATION OUTPUT ---")
    print(output)
    print("---------------------------")

    assert "REJECTED 3 TIMES for same pattern: Mirror Hallucination detected" in output
    assert "Step 1" in output
    assert "Step 5" in output
    print("SUCCESS: Condensation verified.")

if __name__ == "__main__":
    test_condensation()
