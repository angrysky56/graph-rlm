import dataclasses
import sys
import unittest
from types import ModuleType
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

# --- BOILERPLATE MOCKING ---
m_redis = MagicMock()
sys.modules["redis"] = m_redis
sys.modules["falkordb"] = MagicMock()
sys.modules["langchain_community"] = MagicMock()
sys.modules["langchain_community.graphs"] = MagicMock()

# Mock graph_rlm internals
sys.modules["graph_rlm.backend.src.core.config"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.logger"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.guardrails"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.trace"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.semantic_summarizer"] = MagicMock()

from graph_rlm.backend.src.core.db import GraphClient
from graph_rlm.backend.src.core.thimac_memory import (
    ThimacEvent,
    ThimacLevel,
    ThimacMemory,
    ThimacOperation,
)


class TestTLTGDynamics(unittest.TestCase):
    def setUp(self):
        self.memory = ThimacMemory()

    def test_metabolic_oscillation(self):
        """Verify FE calculation and state transitions."""
        # Baseline
        self.assertEqual(self.memory.Pi, 0.2)
        self.assertEqual(self.memory.Rg, 0.8)

        # Ingest a successful thought that triggers ACCEPT (result='')
        thought_data = {
            "id": "t1",
            "prompt": "test",
            "status": "success",
            "compression_gain": 0.5,
        }
        event = self.memory.ingest_thought(thought_data, tool_calls=[])

        # Pi = 0.2 + 0.01 - 0.1 = 0.11
        # Rg = 0.8 + 0.1 (ACCEPT) = 0.9
        # FE = 0.11 + 0.1 = 0.21 -> DELTA
        self.assertAlmostEqual(event.inference_pressure, 0.11)
        self.assertAlmostEqual(event.relational_gravity, 0.9)
        self.assertEqual(event.metabolic_state, "DELTA")

        # Ingest a failed thought with high entropy to push to GAMMA
        thought_data_fail = {
            "id": "t2",
            "prompt": "fail result",
            "status": "failed",
            "compression_gain": 0.0,
        }
        # Clear ACCEPT flags by providing a result
        thought_data_fail["result"] = "error"

        # 4 iterations of failure:
        # Growth per step: 0.05 (fail) + 2*0.05 (tools) = 0.15
        # Rg decrease: 0.05 per fail
        # Start state (after DELTA): Pi=0.11, Rg=0.9
        # 1: Pi=0.26, Rg=0.85 -> FE=0.41 (THETA)
        # 2: Pi=0.41, Rg=0.80 -> FE=0.61 (GAMMA)
        for _ in range(2):
            event = self.memory.ingest_thought(
                thought_data_fail, tool_calls=["t1", "t2"]
            )

        self.assertTrue(event.inference_pressure > 0.4)
        self.assertTrue(event.free_energy > 0.6)
        self.assertEqual(event.metabolic_state, "GAMMA")

    @patch("graph_rlm.backend.src.core.db.GraphClient.query")
    @patch("graph_rlm.backend.src.core.db.GraphClient.find_similar_thoughts")
    def test_resonance_logic(self, mock_find, mock_query):
        """Verify RESONATES_WITH logic in GraphClient."""
        with patch("graph_rlm.backend.src.core.db.FalkorDBGraph"), patch(
            "graph_rlm.backend.src.core.db.settings"
        ):
            client = GraphClient()
            client.query = mock_query
            client.find_similar_thoughts = mock_find
            mock_find.return_value = [{"id": "peer_id", "score": 0.95}]

            client.update_thought_result(
                thought_id="current_id", result="done", embedding=[0.1] * 3072
            )

            resonance_call = False
            for call in mock_query.call_args_list:
                if "RESONATES_WITH" in call[0][0]:
                    resonance_call = True
            self.assertTrue(resonance_call)

    def test_gestalt_thermodynamics(self):
        """Verify thermodynamic metrics in Gestalt string."""
        thought_data = {"id": "t1", "prompt": "test", "status": "success"}
        self.memory.ingest_thought(thought_data)
        gestalt = self.memory.get_gestalt_string()
        self.assertIn("Cog-Metabolism", gestalt)
        self.assertIn("Free Energy ($FE$)", gestalt)


if __name__ == "__main__":
    unittest.main()
