
import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from graph_rlm.backend.src.core.agent import Agent
from graph_rlm.backend.src.core.prompts import build_system_prompt
from graph_rlm.backend.src.core.state import ExecutionState, agent_state

# This test must be run from graph_rlm/backend/src as:
# export PYTHONPATH=$PYTHONPATH:.
# python -m test_dreamer_flow


class TestDreamerAgentFlow(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Prevent actually initializing some things
        with patch("graph_rlm.backend.src.core.agent.AgentRuntime"):
            self.agent = Agent()

        # Mock necessary components
        self.agent.skills_manager = MagicMock()
        self.agent.runtime = MagicMock()
        self.agent.db = MagicMock()
        self.agent.active_repls = {}
        self.agent.morph_memory = MagicMock()

    @patch("graph_rlm.backend.src.core.agent.get_axioms_manager")
    @patch("graph_rlm.backend.src.core.agent.llm")
    async def test_axiom_discovery_pre_flight(self, mock_llm, mock_get_mgr):
        # Setup mocks
        mock_mgr = AsyncMock()
        mock_get_mgr.return_value = mock_mgr
        mock_mgr.find_similar_axioms.return_value = [
            {"name": "test-axiom", "description": "A test validator", "score": 0.9}
        ]

        # Test _initialize_turn
        ctx = await self.agent._initialize_turn(
            prompt="Test prompt",
            parent_id=None,
            session_id="test_session",
            depth=0,
            root_session_id=None,
            turn_id=1,
            recursion_stack=None,
            metadata=None
        )

        # Verify axiom was found and stored
        self.assertIn("relevant_axioms", ctx)
        self.assertEqual(len(ctx["relevant_axioms"]), 1)
        self.assertEqual(ctx["relevant_axioms"][0]["name"], "test-axiom")

    async def test_system_prompt_axiom_injection(self):
        # Test build_system_prompt with relevant_axioms
        relevant_axioms = [{"name": "math-axiom", "description": "Ensures math is correct"}]

        # We need to mock backend_root and rules.md check in prompts.py or just trust the string injection
        with patch("pathlib.Path.exists", return_value=False):
            prompt = await build_system_prompt(
                relevant_axioms=relevant_axioms
            )

        self.assertIn("[RELEVANT AXIOMS (Domain Validators)]", prompt)
        self.assertIn("math-axiom", prompt)
        self.assertIn("Ensures math is correct", prompt)

    @patch("graph_rlm.backend.src.core.agent.Agent._execute_action", new_callable=AsyncMock)
    @patch("graph_rlm.backend.src.core.agent.Agent._generate_thought", new_callable=AsyncMock)
    @patch("graph_rlm.backend.src.core.agent.Agent._initialize_step", new_callable=AsyncMock)
    @patch("graph_rlm.backend.src.core.agent.Agent._initialize_turn", new_callable=AsyncMock)
    @patch("graph_rlm.backend.src.core.agent.Agent._process_response", new_callable=AsyncMock)
    @patch("graph_rlm.backend.src.core.agent.Agent._validate_and_finalize", new_callable=AsyncMock)
    @patch("graph_rlm.backend.src.core.agent.Dreamer", new_callable=MagicMock)
    async def test_reflexion_trigger_on_failure(self, mock_dreamer_cls, mock_validate, mock_process, mock_init_turn, mock_init_step, mock_gen_thought, mock_exec_action):
        # Setup failed execution
        mock_init_turn.return_value = {
            "step": 0, "max_steps": 5, "root_id": "root", "pad": "", "relevant_axioms": []
        }
        mock_gen_thought.return_value = "Run some bad code"
        mock_process.return_value = ("print(1/0)", None)

        # _execute_action returns: output, execution_failed, execution_summary, current_code_hash
        mock_exec_action.return_value = ("ZeroDivisionError", True, "Division by zero", "hash123")

        # Mock Dreamer
        mock_dreamer_inst = AsyncMock()
        mock_dreamer_cls.return_value = mock_dreamer_inst
        mock_dreamer_inst.dream_cycle.return_value = {"insight": "Don't divide by zero!"}

        # To avoid infinite loop, make stop_requested True after 1 step
        self.agent.stop_requested = False

        async def side_effect_stop(*args, **kwargs):
            self.agent.stop_requested = True

        mock_validate.side_effect = side_effect_stop

        # Run query_sync
        await self.agent.query_sync("Trigger failure")

        # Verify Dreamer was called with reflexion_context
        mock_dreamer_inst.dream_cycle.assert_called()
        call_kwargs = mock_dreamer_inst.dream_cycle.call_args.kwargs
        self.assertIn("reflexion_context", call_kwargs)
        self.assertEqual(call_kwargs["reflexion_context"]["error"], "Division by zero")

        # Verify insight was captured
        self.assertEqual(self.agent.last_dream_insight, "Don't divide by zero!")

if __name__ == "__main__":
    unittest.main()
