import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from graph_rlm.backend.src.core.agent import Agent
from graph_rlm.backend.src.core.morphogenesis import MorphologicalMemory


class TestAgentMorphIntegration(unittest.IsolatedAsyncioTestCase):
    @patch("graph_rlm.backend.src.core.agent.llm")
    @patch("graph_rlm.backend.src.core.agent.db")
    async def test_agent_initialization(self, mock_db, mock_llm):
        agent = Agent()
        assert hasattr(agent, "morph_memory")
        assert isinstance(agent.morph_memory, MorphologicalMemory)

    @patch("graph_rlm.backend.src.core.agent.llm")
    @patch("graph_rlm.backend.src.core.agent.db")
    @patch("graph_rlm.backend.src.core.agent.protected_llm_generate")
    @patch("graph_rlm.backend.src.core.agent.scratchpad_builder")
    @patch("graph_rlm.backend.src.core.agent.build_system_prompt")
    @patch("graph_rlm.backend.src.core.agent.dreamer")
    @patch("graph_rlm.backend.src.core.agent.repe")
    async def test_agent_query_sync_seeding(
        self,
        mock_repe,
        mock_dream,
        mock_sys_prompt,
        mock_scratchpad,
        mock_gen,
        mock_db,
        mock_llm,
    ):
        mock_llm.get_embedding = AsyncMock(return_value=[0.1] * 64)
        mock_gen.return_value = "RLM_FINAL_RESPONSE"
        mock_sys_prompt.return_value = "system prompt"
        mock_scratchpad.build_scratchpad.return_value = "mock scratchpad"

        agent = Agent()
        # Mock class methods often called in loop/exit
        agent._generate_validated_response = AsyncMock(return_value="final answer")

        # Seed check
        with patch.object(agent.morph_memory, "seed") as mock_seed:
            await agent.query_sync("test prompt")
            mock_seed.assert_called_once()
            args, _ = mock_seed.call_args
            assert args[0] == [0.1] * 64

    @patch("graph_rlm.backend.src.core.agent.llm")
    @patch("graph_rlm.backend.src.core.agent.db")
    @patch("graph_rlm.backend.src.core.agent.protected_llm_generate")
    @patch("graph_rlm.backend.src.core.agent.scratchpad_builder")
    @patch("graph_rlm.backend.src.core.agent.build_system_prompt")
    @patch("graph_rlm.backend.src.core.agent.dreamer")
    @patch("graph_rlm.backend.src.core.agent.repe")
    async def test_agent_loop_update(
        self,
        mock_repe,
        mock_dream,
        mock_sys_prompt,
        mock_scratchpad,
        mock_gen,
        mock_db,
        mock_llm,
    ):
        mock_llm.get_embedding = AsyncMock(return_value=[0.1] * 64)
        mock_gen.side_effect = ["Step 1 thought", "RLM_FINAL_RESPONSE"]
        mock_sys_prompt.return_value = "system prompt"
        mock_scratchpad.build_scratchpad.return_value = "mock scratchpad"

        agent = Agent()
        agent._generate_validated_response = AsyncMock(return_value="final answer")
        agent._extract_code = MagicMock(return_value="")
        agent.runtime.execute = AsyncMock(return_value=("", False))

        # Update check
        with patch.object(agent.morph_memory, "update") as mock_update:
            with patch.object(agent.morph_memory, "get_gestalt_string") as mock_gestalt:
                mock_gestalt.return_value = "TEST_GESTALT"
                await agent.query_sync("test prompt")
                assert mock_update.call_count >= 1

        # Verify gestalt was passed to scratchpad
        mock_scratchpad.build_scratchpad.assert_called()
        found = False
        for call in mock_scratchpad.build_scratchpad.call_args_list:
            if call.kwargs.get("morph_gestalt") == "TEST_GESTALT":
                found = True
                break
        assert found


if __name__ == "__main__":
    unittest.main()
