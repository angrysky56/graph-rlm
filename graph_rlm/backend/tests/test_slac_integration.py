from unittest.mock import MagicMock, patch

import pytest

from graph_rlm.backend.src.core.reflexion import IntelliSynth
from graph_rlm.backend.src.core.sheaf import SheafMonitor
from graph_rlm.backend.src.core.slac import SLACEngine, TemporalLogicSystem


def test_temporal_logic_consistency():
    # Healthy case
    statements = ["I will check the files.", "I have read the documentation."]
    audit = TemporalLogicSystem.audit_temporal_consistency(statements)
    assert audit["status"] == "STABLE"
    assert audit["tense_span"] == "Cross-Temporal"

    # Inconsistent case (G/F contradicts H/P)
    # "I will start the task" but "Task already done"
    statements = ["I will start the task.", "Task is already done."]
    audit = TemporalLogicSystem.audit_temporal_consistency(statements)
    assert audit["status"] == "INCONSISTENT"
    assert len(audit["contradictions"]) > 0


def test_slac_advancement_scoring():
    engine = SLACEngine(alpha=1.0, beta=1.5)

    # Progress case
    metrics = {"truth": 0.8, "shakiness": 0.2, "improvement": 0.4}
    result = engine.run_cycle(metrics)
    # A(T) = 0.8 + 1.0*(1-0.2) + 1.5*0.4 = 0.8 + 0.8 + 0.6 = 2.2
    assert result["at_score"] == pytest.approx(2.2)
    assert result["status"] == "ADVANCING"
    # Index 5 is RB, Index 6 is M
    assert result["stage"] in ["RB", "M"]


def test_sheaf_temporal_integration():
    monitor = SheafMonitor()

    # Mock data with temporal inconsistency
    hypo = {"id": "hypo_1", "content": "I will run the tests.", "embedding": [1.0] * 32}
    history = [
        MagicMock(
            to_dict=lambda: {
                "id": "hist_1",
                "content": "Tests already completed.",
                "embedding": [1.0] * 32,
                "status": "success",
            }
        )
    ]

    with patch("graph_rlm.backend.src.core.sheaf.db", MagicMock()):
        diag = monitor.diagnose_trace(
            root_id="test", hypothetical_node=hypo, memory_trajectory=history
        )

        # Should trigger LOGICAL_KNOT due to Temporal Inconsistency
        assert diag["status"] == "LOGICAL_KNOT"
        assert "Temporal Inconsistency Detected" in diag["critique"]


@pytest.mark.asyncio
async def test_reflexion_slac_integration():
    # Mock the db import in reflexion.py since it's used inside the class but not at module level
    with patch("graph_rlm.backend.src.core.reflexion.db", MagicMock(), create=True):
        synth = IntelliSynth()

        # Mock metrics to trigger SLAC
        mock_metrics = {
            "shakiness": 0.1,
            "loop_energy": 0.1,
            "topo_status": "consistent",
            "sheaf_score": 0.9,
            "report": "All good",
        }

        with patch.object(
            IntelliSynth, "_gather_metrics", return_value=mock_metrics
        ), patch.object(
            IntelliSynth, "_fetch_genesis", return_value="Genesis"
        ), patch.object(
            IntelliSynth, "_compute_drift", return_value=0.1
        ):

            result = await synth.advancement_cycle(
                trace_context="context",
                current_thought="thought",
                divergence_point="divergence",
            )

            assert "slac_at" in result
            assert "slac_stage" in result
            assert "slac_bar" in result


if __name__ == "__main__":
    pytest.main([__file__])
