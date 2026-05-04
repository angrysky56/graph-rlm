
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Adjust path to backend
backend_path = Path(__file__).parent.parent / "graph_rlm" / "backend" / "src"
sys.path.append(str(backend_path.parent.parent))

# Mock imports that might result in DB connection
sys.modules["graph_rlm.backend.src.core.db"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.llm"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.manager"] = MagicMock()

# Now import agent
# We need to selectively mock the classes inside
from graph_rlm.backend.src.core.agent import Agent


class TestStatelessAgent(unittest.TestCase):
    def setUp(self):
        self.agent = Agent()
        # Mock DB
        self.agent.db = MagicMock()
        # Mock Frontier (The "Wake" Cycle)
        self.agent.db.get_context_frontier.return_value = [
            {"prompt": "Previous Thought 1", "result": "Result 1", "id": "t1"},
            {"prompt": "Previous Thought 2", "result": "Result 2", "id": "t2"}
        ]

        # Mock LLM
        self.agent.llm = MagicMock()
        self.agent.llm.generate.return_value = "This is a thought.\n```python\nprint('Hello World')\n```"
        self.agent.llm.get_embedding.return_value = [0.1] * 768

        # Mock REPL execution
        self.agent.repl_manager = MagicMock()
        self.agent.repl_manager.get_repl.return_value = MagicMock()
        self.agent._execute_code = MagicMock(return_value="Hello World")
        self.agent.active_repls = {"default": "repl_123"}

    def test_query_sync_flow(self):
        print("\n--- Testing Stateless Loop ---")

        # Run query
        response = self.agent.query_sync("Do something", session_id="test_session")

        # 1. Verify "Task" Node Creation
        self.agent.db.create_thought_node.assert_any_call(
            unittest.mock.ANY, "Do something", None, prompt_embedding=None, session_id="test_session", root_session_id="test_session"
        )

        # 2. Verify Wake Cycle (Frontier Call)
        self.agent.db.get_context_frontier.assert_called_with("test_session", limit=5)
        print("[Pass] Wake Cycle: Called get_context_frontier")

        # 3. Verify Code Execution
        self.agent._execute_code.assert_called()
        print("[Pass] Act: Executed Code")

        # 4. Verify Commit (Thought Node for the step)
        # Should be called again for the thought
        # We check that create_thought_node was called at least twice (Task + Step 1)
        self.assertGreaterEqual(self.agent.db.create_thought_node.call_count, 2)
        print("[Pass] Commit: Created nodes for thoughts")

if __name__ == "__main__":
    unittest.main()
