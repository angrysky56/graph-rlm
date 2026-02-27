import asyncio
import sys
from unittest.mock import MagicMock, patch

# --- MOCK DB AND OTHER MODULES BEFORE IMPORTING AGENT ---
# This prevents ConnectionRefusedError during import
mock_db = MagicMock()
sys.modules["graph_rlm.backend.src.core.db"] = MagicMock(db=mock_db)
sys.modules["graph_rlm.backend.src.core.llm"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.dream"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.sheaf"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.omcd"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.navigator"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.thimac_memory"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.scratchpad_builder"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.reflexion"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.rlm_interface"] = MagicMock()
sys.modules["graph_rlm.backend.src.mcp_integration.runtime"] = MagicMock()
sys.modules["graph_rlm.backend.src.mcp_integration.skill_storage"] = MagicMock()

# Now we can import Agent (it will use the mocks)
# Note: we might need to mock more if imports chain
try:
    from graph_rlm.backend.src.core.agent import Agent
    from graph_rlm.backend.src.core.meta_agents import AgentRole, Fragment, meta_agents
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


async def test_synthesizer_role_transition():
    # Setup agent
    # We need to mock more internal attributes that Agent.__init__ might use
    with patch("graph_rlm.backend.src.core.agent.get_skills_manager"), patch(
        "graph_rlm.backend.src.core.agent.ThimacMemory"
    ), patch("graph_rlm.backend.src.core.agent.Navigator"):
        agent = Agent()

    session_id = "test-session"
    root_id = "test-root"

    # Mock Turn Context
    task_profile = {"role": AgentRole.WORKER, "persona": "Worker"}
    turn_ctx = {
        "task_profile": task_profile,
        "exec_state": MagicMock(),
        "root_id": root_id,
    }

    # Mock Meta-Agents Collaboration State
    meta_agents.active_collaborations = {}
    meta_agents.start_collaboration(root_id, "Test task")

    # Simulate High Coherence
    with patch.object(
        meta_agents, "evaluate_coherence", return_value=True
    ), patch.object(
        meta_agents, "get_synthesizer_instructions", return_value="SYNTH INSTRUCTIONS"
    ), patch.object(
        agent, "emit_event"
    ), patch(
        "graph_rlm.backend.src.core.agent.build_system_prompt",
        return_value="System Prompt",
    ):

        # We need to mock _get_dashboard_metrics too
        agent._get_dashboard_metrics = MagicMock(return_value=asyncio.Future())
        agent._get_dashboard_metrics.return_value.set_result({})

        await agent._initialize_step(step=1, session_id=session_id, turn_ctx=turn_ctx)

    # Check if role transitioned
    assert turn_ctx["task_profile"]["role"] == AgentRole.SYNTHESIZER
    assert turn_ctx["task_profile"]["persona"] == "Abstract Synthesizer"

    print("✅ Role transition test passed!")


if __name__ == "__main__":
    asyncio.run(test_synthesizer_role_transition())
