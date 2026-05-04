import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from graph_rlm.backend.src.core.agent import Agent


class TestSelfHealing(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Mock dependencies to avoid real network/DB calls
        self.mock_db = MagicMock()
        self.mock_llm = MagicMock()
        self.mock_llm.generate = AsyncMock()
        self.mock_llm.get_embedding = AsyncMock(return_value=[0.1] * 1536)
        self.mock_llm.config = {"model": "test-model"}

        self.patcher_db = patch("graph_rlm.backend.src.core.agent.db", self.mock_db)
        self.patcher_llm = patch("graph_rlm.backend.src.core.agent.llm", self.mock_llm)
        self.patcher_skills = patch(
            "graph_rlm.backend.src.core.agent.get_skills_manager"
        )
        self.patcher_nav = patch("graph_rlm.backend.src.core.agent.Navigator")
        self.patcher_sheaf = patch("graph_rlm.backend.src.core.agent.sheaf")
        self.patcher_dreamer = patch("graph_rlm.backend.src.core.agent.dreamer")
        self.patcher_meta = patch("graph_rlm.backend.src.core.meta_agents.meta_agents")
        self.patcher_protected = patch(
            "graph_rlm.backend.src.core.agent.protected_llm_generate"
        )

        self.mock_sheaf = self.patcher_sheaf.start()
        self.mock_dreamer = self.patcher_dreamer.start()
        self.mock_meta = self.patcher_meta.start()
        self.mock_protected = self.patcher_protected.start()
        self.patcher_db.start()
        self.patcher_llm.start()
        self.patcher_skills.start()
        self.patcher_nav.start()

        # Mock meta_agents functions
        self.mock_meta.generate_sub_agent_profile = AsyncMock(
            return_value={"persona": "test", "tools": []}
        )
        self.mock_meta.get_worker_instructions = MagicMock(return_value="instructions")
        self.mock_protected.return_value = "I am done. RLM_FINAL_OUTPUT: Done."

        self.agent = Agent()
        self.agent.db = self.mock_db
        self.agent.llm = self.mock_llm

    async def asyncTearDown(self):
        patch.stopall()

    @patch(
        "graph_rlm.backend.src.core.agent.build_system_prompt",
        AsyncMock(return_value="System Prompt"),
    )
    async def test_rejection_recovery_cycle(self):
        """Verify that a Dreamer rejection correctly updates state and awaits cycle."""
        session_id = "test-session"
        prompt = "test task"

        # 1. Mock LLM to return a "Done" response
        self.mock_llm.generate.return_value = "I am done. RLM_FINAL_OUTPUT: Done."

        # 2. Mock Dreamer to Reject first, then validate (not needed for loop, but for completeness)
        rejection_val = {
            "status": "invalid",
            "instruction": "Fix the hallucination",
            "reasons": ["Test reason"],
        }
        self.mock_dreamer.validate_response = AsyncMock(
            side_effect=[rejection_val, {"status": "valid"}]
        )
        self.mock_dreamer.dream_cycle = AsyncMock(return_value={"status": "lucid"})

        # 3. Mock Sheaf to be healthy
        self.mock_sheaf.check_axiomatic_consistency = AsyncMock(
            return_value={"status": "OK"}
        )
        self.mock_sheaf.diagnose_trace = MagicMock(
            return_value={"status": "HEALTHY", "consistency_energy": 0.0}
        )

        # 4. Run one step of the query loop (or a limited mock query)
        # We'll mock the internal _refresh_scratchpad to avoid DB complexity
        self.agent._refresh_scratchpad = AsyncMock(return_value="Scratchpad")
        self.agent._verify_epistemic_integrity = MagicMock(
            return_value={"status": "OK"}
        )
        self.agent._generate_axiom_search_query = AsyncMock(return_value="query")

        # Run the loop with a timeout/limit
        # We want to see if it continues after rejection
        # To avoid infinite loop in test, we'll patch the while condition or just check the first rejection flow

        # Mock execute_code to return success
        self.agent._execute_code = AsyncMock(return_value=("output", False, None))

        # We need to stop the loop manually or mock max_steps
        with patch.object(self.agent, "emit_event") as mock_emit:
            # Trigger the loop
            print("Starting query_sync task...")
            task_task = asyncio.create_task(
                self.agent.query_sync(prompt, session_id=session_id)
            )

            # Wait for some time for the first rejection to process
            print("Waiting for rejection...")
            for _ in range(20):
                await asyncio.sleep(0.1)
                if self.agent.last_dream_insight:
                    break

            print(f"Agent last_dream_insight: {self.agent.last_dream_insight}")
            print(f"Awaiting validation: {self.agent.awaiting_validation}")
            print(
                f"Validate response call count: {self.mock_dreamer.validate_response.call_count}"
            )

            # Verifications
            self.assertIsNotNone(
                self.agent.last_dream_insight,
                "last_dream_insight should have been set after rejection",
            )
            self.assertEqual(
                self.agent.last_dream_insight,
                "Fix the hallucination. REASONS: Test reason",
            )
            self.assertFalse(self.agent.awaiting_validation)
            self.assertFalse(self.agent.synthesis_triggered)

            # Verify dream_cycle was awaited (not backgrounded)
            self.mock_dreamer.dream_cycle.assert_awaited()

            # Allow the loop to finish (it should find "valid" second time and break)
            print("Waiting for task to complete...")
            await asyncio.wait_for(task_task, timeout=5)

            self.assertTrue(self.agent._final_output_emitted)


if __name__ == "__main__":
    unittest.main()
