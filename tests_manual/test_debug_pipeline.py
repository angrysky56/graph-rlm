import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Adjust path to include repo root
sys.path.append(str(Path(__file__).parents[1]))

from graph_rlm.backend.src.core.rlm_interface import RLMInterface
from graph_rlm.backend.src.core.scratchpad_builder import ScratchpadBuilder


class TestPipelineDebug(unittest.TestCase):
    def setUp(self):
        self.builder = ScratchpadBuilder()
        self.mock_agent = MagicMock()
        self.mock_agent.project_root = Path("/tmp")
        self.mock_agent.stop_requested = False
        self.mock_agent.global_stop_event.is_set.return_value = False
        self.rlm = RLMInterface(self.mock_agent, "sess_1", "root_1")

    def test_scratchpad_blindness_fix(self):
        """Verify that 'Discovery' steps are NOT condensed even if old."""
        # Create 15 steps. Step 1 is discovery.
        # Steps 0-4 are "old" if window is 10 and total is 15.

        # Mock data output from DB query
        rows = []
        for i in range(15):
            prompt = f"Step {i}"
            if i == 1:
                prompt = "await rlm.describe_tools('mcp.brave_search')"
            elif i == 2:
                prompt = "print(dir(mcp))"

            rows.append(
                {
                    "id": f"id_{i}",
                    "prompt": prompt,
                    "status": "success",
                    "result": "Some result",
                    "created_at": 1000 + i,
                    "repl_id": "repl_1",
                    "execution_summary": None,
                    "next_action": None,
                    "dreamer_analysis": None,
                    "final_response": None,
                    "turn_id": 1,
                    "step_id": i,
                    "code_hash": "abc",
                }
            )

        output = self.builder._format_progress_rows(rows)

        # Check Step 1 (describe_tools) - Should NOT be condensed
        self.assertIn("describe_tools", output)
        self.assertNotIn("Step 1 [Detail integrated", output)

        # Check Step 0 (normal) - Should be condensed (it is > 10 steps ago)
        # Wait, 15 steps total. i=0 is < 15-10=5. So 0,1,2,3,4 are candidates.
        # Step 0 should be condensed.
        # But wait, my logic was: is_old_step = i < (total_count - 10)
        # 0 < 5 -> True.
        # So "Step 0" prompt should NOT appear fully if condensed?
        # The condensed format says "[Detail integrated...]"

        # Let's check regex for Step 0
        if "Step 0" in output:
            # If "Step 0" text is in output, it might be the ID S0, not the prompt "Step 0"
            pass

        # We look for the "integrated" message for step 0
        lines = output.split("\n")
        step_0_line = next((line for line in lines if "S0" in line), None)
        self.assertIsNotNone(step_0_line)
        # Actually step_id=0 -> S0.
        # The prompt "Step 0" would be replaced by logic.

        # Check Step 1 specifically
        step_1_line = next((line for line in lines if "S1" in line), "")
        print(f"Step 1 Line: {step_1_line}")
        self.assertNotIn("integrated in Morphological Gestalt", step_1_line)
        self.assertIn("describe_tools", step_1_line)

    def test_describe_tools(self):
        """Verify describe_tools returns docs for a valid module."""
        # We'll use unittest.mock to mock importlib to avoid needing real deps
        with patch("importlib.import_module") as mock_import:
            mock_mod = MagicMock()

            def tool_func(x):
                """Test Tool Docstring."""
                pass

            mock_mod.test_tool = tool_func
            mock_import.return_value = mock_mod

            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.rlm.describe_tools("mcp.test_module"))

            print(f"Describe Result:\n{result}")
            self.assertIn("## Tools in 'test_module'", result)
            self.assertIn("**test_tool(x)**", result)
            self.assertIn("Test Tool Docstring", result)


if __name__ == "__main__":
    unittest.main()
