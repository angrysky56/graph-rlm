import sys
import os
import unittest
from unittest.mock import MagicMock

# Adjust path to include backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.agent import Agent

class TestAgent(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.agent = Agent()
        self.agent.db = MagicMock()
        self.agent.llm = MagicMock()

        # Real REPL manager (we want to test REPL integration)
        # self.agent.repl_manager is already real

    def test_query_with_repl(self):
        # Setup LLM to return code
        self.agent.llm.generate.return_value = (
            "I will calculate this.\n"
            "```python\n"
            "x = 5\n"
            "y = 10\n"
            "result = x * y\n"
            "print(result)\n"
            "```"
        )

        # Run query
        response = self.agent.query("Calculate 5 * 10")

        # Verify DB called
        self.agent.db.create_thought_node.assert_called_once()
        self.agent.db.update_thought_result.assert_called_once()

        # Verify Response contains execution result
        self.assertIn("[REPL Output]", response)
        self.assertIn("50", response)

    def test_recursive_primitive_exposed(self):
        # Setup LLM to return code that calls rlm.query
        # We need to mock agent.query to avoid infinite recursion or complex mocking
        # But we want to test that 'rlm' object exists and calls back.

        # We'll use a side_effect on the inner query to verify it was called.
        self.agent.llm.generate.return_value = (
            "Recursive call:\n"
            "```python\n"
            "rlm.query('child task')\n"
            "```"
        )

        # Mock the *recursive* call logic?
        # rlm.query calls self.agent.query.
        # We can wrap self.agent.query with a mock EXCEPT for the first call?
        # Easier: Spy on it.

        original_query = self.agent.query
        self.agent.query = MagicMock(side_effect=original_query)

        # We need to handle the *second* call to LLM (for child task)
        # return_value can be a list (iterator) implies multiple calls
        self.agent.llm.generate.side_effect = [
            # 1. First call returns code to call rlm.query
             "Recursive call:\n```python\nrlm.query('child task')\n```",
            # 2. Second call (nested) returns simple answer
             "Child task done."
        ]

        # Execute
        final_response = self.agent.query("Start recursion")

        # Verify agent.query was called twice: 1. "Start recursion", 2. "child task"
        # The first call is the one we invoked manually.
        # The second call is from inside the REPL.
        call_args_list = self.agent.query.call_args_list
        # Note: self.agent.query is the Mock wrapper.

        # Check arguments
        self.assertEqual(len(call_args_list), 2)
        self.assertEqual(call_args_list[0][0][0], "Start recursion")
        self.assertEqual(call_args_list[1][0][0], "child task")

        # Restore (good practice)
        self.agent.query = original_query

if __name__ == '__main__':
    unittest.main()
