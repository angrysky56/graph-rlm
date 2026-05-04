import asyncio
import importlib
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path if needed for relative imports to work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Create mocks for the dependencies
mock_agent_mod = MagicMock()
mock_agent_mod.agent = MagicMock()
mock_agent_mod.agent.query_sync = AsyncMock(return_value="Mock Agent Result")

mock_db_mod = MagicMock()
mock_db_mod.db = MagicMock()
# Mock query to return a list (safe default)
mock_db_mod.db.query.return_value = [{"count": 0, "n.status": "consolidated"}]
mock_db_mod.db.create_thought_node = MagicMock()

mock_dreamer_mod = MagicMock()
mock_dreamer_mod.dreamer = MagicMock()
mock_dreamer_mod.dreamer.dream_cycle = AsyncMock(
    return_value={"status": "lucid", "insight": "Mock Insight"}
)

mock_llm_mod = MagicMock()
mock_llm_mod.llm = MagicMock()
mock_llm_mod.llm.get_embedding = AsyncMock(return_value=[0.1] * 1536)
mock_llm_mod.llm.generate = AsyncMock(return_value="ACK")

mock_repe_mod = MagicMock()
mock_repe_mod.repe = MagicMock()
mock_repe_mod.repe.calibrate = AsyncMock()
mock_repe_mod.repe.scan_thought = MagicMock(return_value={"Shakiness": -0.5})

mock_sheaf_mod = MagicMock()
mock_sheaf_mod.sheaf = MagicMock()
mock_sheaf_mod.sheaf.diagnose_trace = MagicMock(return_value={"status": "LOGICAL_KNOT"})

# Mock the scratchpad builder which is imported inside functions
mock_scratchpad_builder = MagicMock()
mock_scratchpad_builder.build_scratchpad.return_value = "Mock Context"


class TestCLIPipeline(unittest.TestCase):

    def setUp(self):
        # Reset mocks before each test
        mock_agent_mod.agent.query_sync.reset_mock()
        mock_db_mod.db.query.reset_mock()
        mock_db_mod.db.create_thought_node.reset_mock()
        mock_dreamer_mod.dreamer.dream_cycle.reset_mock()
        mock_llm_mod.llm.get_embedding.reset_mock()
        mock_llm_mod.llm.generate.reset_mock()
        mock_repe_mod.repe.calibrate.reset_mock()
        mock_repe_mod.repe.scan_thought.reset_mock()
        mock_sheaf_mod.sheaf.diagnose_trace.reset_mock()
        mock_scratchpad_builder.build_scratchpad.reset_mock()

    @patch.dict(
        sys.modules,
        {
            "graph_rlm.backend.src.core.agent": mock_agent_mod,
            "graph_rlm.backend.src.core.db": mock_db_mod,
            "graph_rlm.backend.src.core.dream": mock_dreamer_mod,
            "graph_rlm.backend.src.core.llm": mock_llm_mod,
            "graph_rlm.backend.src.core.repe": mock_repe_mod,
            "graph_rlm.backend.src.core.sheaf": mock_sheaf_mod,
            "graph_rlm.backend.src.core.scratchpad_builder": MagicMock(
                scratchpad_builder=mock_scratchpad_builder
            ),
        },
    )
    def test_repe_check(self):
        """Test the --check repe logic"""
        # We must import inside the patched context
        if "graph_rlm.backend.src.cli" in sys.modules:
            del sys.modules["graph_rlm.backend.src.cli"]

        # Also need to mock sys.argv to avoid argparse issues if run directly
        with patch.object(sys, "argv", ["cli.py", "--check", "repe"]):
            from graph_rlm.backend.src import cli

            asyncio.run(cli.test_live_repe())

        # Verify interactions
        mock_repe_mod.repe.calibrate.assert_called_once()
        mock_llm_mod.llm.get_embedding.assert_called()
        mock_repe_mod.repe.scan_thought.assert_called()

    @patch.dict(
        sys.modules,
        {
            "graph_rlm.backend.src.core.agent": mock_agent_mod,
            "graph_rlm.backend.src.core.db": mock_db_mod,
            "graph_rlm.backend.src.core.dream": mock_dreamer_mod,
            "graph_rlm.backend.src.core.llm": mock_llm_mod,
            "graph_rlm.backend.src.core.repe": mock_repe_mod,
            "graph_rlm.backend.src.core.sheaf": mock_sheaf_mod,
        },
    )
    def test_sheaf_check(self):
        """Test the --check sheaf logic"""
        if "graph_rlm.backend.src.cli" in sys.modules:
            del sys.modules["graph_rlm.backend.src.cli"]

        with patch.object(sys, "argv", ["cli.py", "--check", "sheaf"]):
            from graph_rlm.backend.src import cli

            asyncio.run(cli.test_live_sheaf())

        # Verify interactions
        self.assertTrue(mock_db_mod.db.query.called)
        self.assertTrue(mock_db_mod.db.create_thought_node.called)
        self.assertTrue(mock_sheaf_mod.sheaf.diagnose_trace.called)

    @patch.dict(
        sys.modules,
        {
            "graph_rlm.backend.src.core.agent": mock_agent_mod,
            "graph_rlm.backend.src.core.db": mock_db_mod,
            "graph_rlm.backend.src.core.dream": mock_dreamer_mod,
            "graph_rlm.backend.src.core.llm": mock_llm_mod,
            "graph_rlm.backend.src.core.repe": mock_repe_mod,
            "graph_rlm.backend.src.core.sheaf": mock_sheaf_mod,
            # Mock scratchpad builder specifically for this test
            "graph_rlm.backend.src.core.scratchpad_builder": MagicMock(
                scratchpad_builder=mock_scratchpad_builder
            ),
        },
    )
    def test_dreamer_check(self):
        """Test the --check dreamer logic"""
        if "graph_rlm.backend.src.cli" in sys.modules:
            del sys.modules["graph_rlm.backend.src.cli"]

        with patch.object(sys, "argv", ["cli.py", "--check", "dreamer"]):
            from graph_rlm.backend.src import cli

            asyncio.run(cli.test_live_dreamer())

        # Verify interactions
        self.assertTrue(mock_db_mod.db.create_thought_node.called)
        self.assertTrue(mock_scratchpad_builder.build_scratchpad.called)
        self.assertTrue(mock_dreamer_mod.dreamer.dream_cycle.called)

        # Check args passed to dream_cycle
        call_kwargs = mock_dreamer_mod.dreamer.dream_cycle.call_args.kwargs
        # Depending on how the mock captures it, it might be in args or kwargs
        if not call_kwargs:
            # Check positional args if any
            pass
        else:
            self.assertEqual(call_kwargs.get("context"), "Mock Context")

    @patch.dict(
        sys.modules,
        {
            "graph_rlm.backend.src.core.agent": mock_agent_mod,
            "graph_rlm.backend.src.core.db": mock_db_mod,
            "graph_rlm.backend.src.core.dream": mock_dreamer_mod,
            "graph_rlm.backend.src.core.llm": mock_llm_mod,
            "graph_rlm.backend.src.core.repe": mock_repe_mod,
            "graph_rlm.backend.src.core.sheaf": mock_sheaf_mod,
        },
    )
    def test_agent_check(self):
        """Test the --check agent logic"""
        if "graph_rlm.backend.src.cli" in sys.modules:
            del sys.modules["graph_rlm.backend.src.cli"]

        with patch.object(
            sys, "argv", ["cli.py", "--check", "agent", "--prompt", "Test Prompt"]
        ):
            from graph_rlm.backend.src import cli

            asyncio.run(cli.test_live_agent(custom_prompt="Test Prompt"))

        # Verify interactions
        mock_agent_mod.agent.query_sync.assert_called_once()
        call_kwargs = mock_agent_mod.agent.query_sync.call_args.kwargs
        self.assertEqual(call_kwargs.get("prompt"), "Test Prompt")

    @patch.dict(
        sys.modules,
        {
            "graph_rlm.backend.src.core.agent": mock_agent_mod,
            "graph_rlm.backend.src.core.db": mock_db_mod,
            "graph_rlm.backend.src.core.dream": mock_dreamer_mod,
            "graph_rlm.backend.src.core.llm": mock_llm_mod,
            "graph_rlm.backend.src.core.repe": mock_repe_mod,
            "graph_rlm.backend.src.core.sheaf": mock_sheaf_mod,
        },
    )
    def test_llm_debug(self):
        """Test the --llm-debug reasoning probe"""
        if "graph_rlm.backend.src.cli" in sys.modules:
            del sys.modules["graph_rlm.backend.src.cli"]

        with patch.object(sys, "argv", ["cli.py", "--llm-debug"]):
            from graph_rlm.backend.src import cli

            asyncio.run(cli.llm_debug_test())

        # Verify interactions
        mock_llm_mod.llm.generate.assert_called_once()
        args, _ = mock_llm_mod.llm.generate.call_args
        self.assertIn("Diagnostic Test", args[0])


if __name__ == "__main__":
    unittest.main()
