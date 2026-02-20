import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add repo root to sys.path
repo_root = Path("/home/ty/Repositories/ai_workspace/graph-rlm")
sys.path.insert(0, str(repo_root))

from graph_rlm.backend.src.core.rlm_interface import RLMInterface


def test_list_skills_execution():
    print("Setting up mock agent...")
    mock_agent = MagicMock()
    mock_agent.execution_logs = {}

    print("Instantiating RLMInterface...")
    rlm = RLMInterface(mock_agent, "session-123", "root-123")

    print("Calling list_skills()...")
    try:
        skills = rlm.list_skills()
        print(f"SUCCESS: list_skills returned: {type(skills)}")
        # It might return error dict if skills system is not available, which is fine for this test
        # as we are testing if the method runs, not if DB is connected.
        print(f"Result: {skills}")
    except Exception as e:
        print(f"FAILURE: list_skills raised exception: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_list_skills_execution()
