from unittest.mock import MagicMock, patch

import pytest

from graph_rlm.backend.src.core.meta_agents import (
    AgentRole,
    MetaAgentController,
    SubAgentProfile,
)


@pytest.mark.asyncio
async def test_generate_sub_agent_profile_logging_lazy():
    # Setup controller and mocks
    controller = MetaAgentController()

    # Mock llm.generate_structured to return a valid profile
    mock_profile = SubAgentProfile(
        persona="Test Persona",
        tools=["rlm"],
        reasoning="Test Reasoning"
    )

    with patch("graph_rlm.backend.src.core.meta_agents.llm") as mock_llm, \
         patch("graph_rlm.backend.src.core.meta_agents.logger") as mock_logger:

        # 1. Test successful case logging (logger.info at line 169)
        # We need should_spawn_breakers to be True
        controller.should_spawn_breakers("complex task analyze", 6000)

        # Check if logger.info was called with lazy formatting
        # info(msg, *args) -> line 169: logger.info("[MetaAgent] ...", context_size, has_complexity)
        mock_logger.info.assert_called_with(
            "[MetaAgent] Breaker spawn recommended: context=%d, has_complexity=%s",
            6000,
            True
        )

        # 2. Test failed skill discovery logging (logger.warning at line 262)
        mock_skills = MagicMock()
        mock_skills.find_similar_skills.side_effect = Exception("Skill Error")

        await controller.generate_sub_agent_profile("test task", skills_manager=mock_skills)

        # Check if logger.warning was called with lazy formatting
        # Expected: logger.warning("Skill discovery failed...", task[:50], e, exc_info=True)
        # CURRENT (Buggy): logger.warning(f"Skill discovery failed... -> {e}", exc_info=True)

        # This assertion is expected to FAIL if it's currently using f-strings
        # Because mock_logger.warning.assert_called_with("format", arg1, arg2, exc_info=True)
        # won't match a single string.

        # Note: We need to match what the REFACTOR will look like:
        # logger.warning("Skill discovery failed during profiling for task: %s... -> %s", task[:50], e, exc_info=True)

        # Let's just check the first argument isn't an f-string (hard to do with assert_called_with)
        # but we can inspect call_args.

        warning_call = [call for call in mock_logger.warning.call_args_list if "Skill discovery failed" in str(call)]
        assert len(warning_call) > 0
        args, kwargs = warning_call[0]
        assert len(args) > 1, f"Expected lazy formatting (multiple args), but got: {args}"
        assert "%s" in args[0], "Expected format string with % placeholders"

@pytest.mark.asyncio
async def test_generate_sub_agent_profile_llm_failure_logging_lazy():
    controller = MetaAgentController()

    with patch("graph_rlm.backend.src.core.meta_agents.llm") as mock_llm, \
         patch("graph_rlm.backend.src.core.meta_agents.logger") as mock_logger:

        mock_llm.generate_structured.side_effect = Exception("LLM Error")

        await controller.generate_sub_agent_profile("test task")

        # 3. Test failed LLM profiling logging (logger.error at line 305)
        # Expected: logger.error("LLM Profiling failed...", task[:50], e, exc_info=True)
        # CURRENT (Buggy): logger.error(f"LLM Profiling failed... -> {e}", exc_info=True)

        error_calls = [call for call in mock_logger.error.call_args_list if "LLM Profiling failed" in str(call)]
        assert len(error_calls) > 0
        args, kwargs = error_calls[0]
        assert len(args) > 1, f"Expected lazy formatting (multiple args), but got: {args}"
        assert "%s" in args[0], "Expected format string with % placeholders"
