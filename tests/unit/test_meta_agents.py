import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from graph_rlm.backend.src.core.meta_agents import (
    MetaAgentController,
    AgentRole,
    Fragment,
    CollaborationState,
    SubAgentProfile,
    Operator,
)

class TestMetaAgentController:
    @pytest.fixture
    def controller(self):
        with patch("graph_rlm.backend.src.core.meta_agents.db") as mock_db:
            return MetaAgentController()

    def test_start_collaboration(self, controller):
        state = controller.start_collaboration("root-1", "test task")
        assert state.root_session_id == "root-1"
        assert state.task == "test task"
        assert "root-1" in controller.active_collaborations

    def test_get_collaboration(self, controller):
        controller.start_collaboration("root-1", "test task")
        state = controller.get_collaboration("root-1")
        assert state is not None
        assert state.task == "test task"
        
        assert controller.get_collaboration("missing") is None

    def test_should_spawn_breakers(self, controller):
        # Small task, small context -> False
        assert controller.should_spawn_breakers("do stuff", 100) is False
        
        # Complexity keyword -> True
        assert controller.should_spawn_breakers("analyze this", 100) is True
        
        # Large context -> True
        assert controller.should_spawn_breakers("do stuff", 6000) is True
        
        # Max depth reached -> False
        assert controller.should_spawn_breakers("analyze this", 6000, depth=3) is False

    @pytest.mark.asyncio
    async def test_generate_sub_agent_profile_success(self, controller):
        mock_llm = MagicMock()
        mock_profile = SubAgentProfile(
            persona="Specialist",
            tools=["rlm.recall", "mcp.search"],
            reasoning="Reasoning"
        )
        mock_llm.generate_structured = AsyncMock(return_value=mock_profile)
        
        with patch("graph_rlm.backend.src.core.meta_agents.llm", mock_llm):
            profile = await controller.generate_sub_agent_profile("task", mcp_names=["search"])
            
            assert profile["persona"] == "Specialist"
            assert "rlm.recall" in profile["tools"]
            mock_llm.generate_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_sub_agent_profile_with_skills(self, controller):
        mock_skills = MagicMock()
        mock_skills.find_similar_skills = AsyncMock(return_value=[
            {"name": "Skill1", "function_name": "func1", "description": "desc1"}
        ])
        
        mock_llm = MagicMock()
        mock_profile = SubAgentProfile(
            persona="Skill Specialist",
            tools=["skills.Skill1"],
            reasoning="Reasoning"
        )
        mock_llm.generate_structured = AsyncMock(return_value=mock_profile)
        
        with patch("graph_rlm.backend.src.core.meta_agents.llm", mock_llm):
            profile = await controller.generate_sub_agent_profile("task", skills_manager=mock_skills)
            
            assert profile["persona"] == "Skill Specialist"
            assert "skills.Skill1" in profile["tools"]
            mock_skills.find_similar_skills.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_sub_agent_profile_llm_failure(self, controller):
        mock_llm = MagicMock()
        mock_llm.generate_structured = AsyncMock(side_effect=Exception("LLM Fail"))
        
        with patch("graph_rlm.backend.src.core.meta_agents.llm", mock_llm):
            # Should return default profile on failure
            profile = await controller.generate_sub_agent_profile("task")
            # The fallback in code returns "Autonomous Generalist" or similar
            assert "Generalist" in profile["persona"]
