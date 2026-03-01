from unittest.mock import MagicMock, patch

import pytest

from graph_rlm.backend.src.core.sheaf import SheafMonitor
from graph_rlm.backend.src.core.thimac_memory import (
    ThimacEvent,
    ThimacLevel,
    ThimacMemory,
    ThimacOperation,
)


@pytest.fixture
def sheaf():
    return SheafMonitor()


@pytest.fixture
def memory():
    return ThimacMemory()


def test_sheaf_nars_mapping(sheaf):
    # Create a "healthy" mock trajectory with high similarity to global sections
    embedding = [1.0] * 32  # Mock embedding
    hypo = {"id": "hypo_1", "embedding": embedding, "content": "test"}

    # Mock history with the same embedding (max similarity = 1.0)
    history = [
        ThimacEvent(
            thought_id="1",
            operation=ThimacOperation.PROCESS,
            level=ThimacLevel.EXISTENCE,
            status="success",
            embedding=embedding,
            session_id="test",
            root_session_id="test",
            logical_id="1",
        )
    ]

    diag = sheaf.diagnose_trace(
        root_id="test", hypothetical_node=hypo, memory_trajectory=history
    )

    # Assert NARS mapping
    assert diag["nars_f"] == pytest.approx(1.0)
    assert diag["nars_c"] > 0.8


def test_sheaf_contradiction_low_confidence(sheaf):
    # Simulate high energy (contradiction)
    with patch.object(
        SheafMonitor,
        "calculate_h1_obstruction",
        return_value={"score": 0.8, "rationale": "knot"},
    ):
        embedding = [1.0, 0.0] * 16
        hypo = {"id": "hypo_contra", "embedding": embedding, "content": "contradiction"}
        history = [
            ThimacEvent(
                thought_id="1",
                operation=ThimacOperation.PROCESS,
                level=ThimacLevel.EXISTENCE,
                status="success",
                embedding=embedding,
                session_id="test",
                root_session_id="test",
                logical_id="1",
            )
        ]

        diag = sheaf.diagnose_trace(
            root_id="test", hypothetical_node=hypo, memory_trajectory=history
        )

        assert diag["nars_c"] == pytest.approx(1.0 / (1.0 + 0.8))


@pytest.mark.asyncio
async def test_agent_contradiction_trigger():
    from graph_rlm.backend.src.core.agent import Agent

    # Mock dependencies in the module before Agent is used
    with patch("graph_rlm.backend.src.core.agent.db", MagicMock()), patch(
        "graph_rlm.backend.src.core.agent.llm", MagicMock()
    ):

        # Agent() takes no arguments in __init__
        agent = Agent()

        # Manually inject or mock morph_memory state
        agent.morph_memory = ThimacMemory()
        event = ThimacEvent(
            thought_id="1",
            operation=ThimacOperation.PROCESS,
            level=ThimacLevel.EXISTENCE,
            status="success",
            session_id="test",
            sheaf_score=0.85,
        )
        agent.morph_memory.store(event)

        turn_ctx = {
            "task_profile": {"role": "explorer"},
            "exec_state": MagicMock(),
            "relevant_axioms": [],
        }

        # This will trigger initialize_step which checks dashboard_data -> morph_memory
        await agent._initialize_step(step=1, session_id="test", turn_ctx=turn_ctx)

        system_prompt = turn_ctx["system_prompt"]
        assert "TOPOLOGICAL DEFECT DETECTED" in system_prompt
        assert "await rlm.query(...)" in system_prompt


if __name__ == "__main__":
    pytest.main([__file__])
