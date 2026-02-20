import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock

# 1. SETUP PATHS
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 2. PRE-IMPORT MOCKING (CRITICAL)
# We must mock external dependencies BEFORE importing Dreamer to avoid
# connection attempts and missing packages at import time.

# Mock pydantic_ai (not installed in test env)
sys.modules["pydantic_ai"] = MagicMock()

mock_db_module = MagicMock()
mock_db_module.db = MagicMock()
mock_db_module.GraphClient = MagicMock
sys.modules["graph_rlm.backend.src.core.db"] = mock_db_module

mock_sheaf_module = MagicMock()
mock_sheaf_module.sheaf = MagicMock()
sys.modules["graph_rlm.backend.src.core.sheaf"] = mock_sheaf_module

mock_repe_module = MagicMock()
mock_repe_module.repe = MagicMock()
sys.modules["graph_rlm.backend.src.core.repe"] = mock_repe_module

mock_omcd_module = MagicMock()
mock_omcd_module.omcd = MagicMock()
sys.modules["graph_rlm.backend.src.core.omcd"] = mock_omcd_module

mock_reflexion_module = MagicMock()
mock_reflexion_module.intelli_synth = AsyncMock()
sys.modules["graph_rlm.backend.src.core.reflexion"] = mock_reflexion_module

# Mock remaining import chain dependencies
sys.modules["graph_rlm.backend.src.core.llm"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.navigator"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.core"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.config"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.trace"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.logger"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.services"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.services.circuit"] = MagicMock()
sys.modules["graph_rlm.backend.src.mcp_integration"] = MagicMock()
sys.modules["graph_rlm.backend.src.mcp_integration.skill_storage"] = MagicMock()

# 3. NOW IMPORT CLASSES  # noqa: E402
from graph_rlm.backend.src.core.dream import Dreamer  # noqa: E402


# --- Helper: Build a mock LLM judgment JSON string ---
def _llm_judgment(verdict="valid", confidence=0.8, reasons=None, instruction=""):
    """Return a JSON string matching the Dreamer LLM classification format."""
    return json.dumps(
        {
            "verdict": verdict,
            "confidence": confidence,
            "reasons": reasons or [],
            "instruction": instruction,
        }
    )


class TestValidationProtocol(unittest.IsolatedAsyncioTestCase):
    """
    TDD Contract for Agent-Dreamer Validation Protocol (v3).
    Validates LLM-driven classification over organized metrics.
    """

    async def asyncSetUp(self):
        self.dreamer = Dreamer()
        # Reset mocks between tests
        mock_sheaf_module.sheaf.reset_mock()
        mock_repe_module.repe.reset_mock()
        mock_omcd_module.omcd.reset_mock()
        mock_reflexion_module.intelli_synth.reset_mock()

        # Mock llm — both get_embedding (vector) and generate (LLM classification)
        self.dreamer.llm = MagicMock()
        self.dreamer.llm.get_embedding = AsyncMock(return_value=[0.1] * 128)
        # Default: LLM returns "valid" judgment
        self.dreamer.llm.generate = AsyncMock(return_value=_llm_judgment("valid", 0.85))

        # Mock db.query — _get_session_trace makes TWO queries:
        # 1. Aggregate metrics  2. Recent nodes with timestamps
        self.dreamer.db = MagicMock()
        self.dreamer.db.query = MagicMock(
            side_effect=[
                # First call: aggregate metrics
                [
                    {
                        "step_count": 3,
                        "turns": [1, 2],
                        "repls": ["repl-1"],
                        "failures": 0,
                    }
                ],
                # Second call: recent nodes with timestamps
                [
                    {
                        "id": "node-a",
                        "status": "completed",
                        "ts": "2026-02-14T12:00:00",
                        "repl_id": "repl-1",
                    },
                    {
                        "id": "node-b",
                        "status": "thinking",
                        "ts": "2026-02-14T11:59:00",
                        "repl_id": "repl-1",
                    },
                ],
            ]
        )

        # Configure default subsystem mocks
        mock_repe_module.repe.scan_thought.return_value = {
            "Shakiness": 0.5,
            "Confluence": 0.3,
            "Evasion": 0.1,
            "Freedom": 0.4,
        }
        mock_sheaf_module.sheaf.diagnose_trace.return_value = {
            "status": "HEALTHY",
            "energy": 0.1,
            "confidence": 0.8,
        }
        mock_omcd_module.omcd.evaluate_step.return_value = {
            "should_stop": False,
            "q_stop": 0.3,
            "threshold": 0.6,
        }

    async def test_happy_path_validation(self):
        """TC1: Valid response passes all checks → RLM_DREAMER_VALIDATED."""
        candidate = "The answer is 42."
        context = "Did deep thought."

        result = await self.dreamer.validate_response(candidate, context)

        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["event"], "RLM_DREAMER_VALIDATED")

    async def test_diagnose_trace_receives_edges(self):
        """TC2: diagnose_trace is called WITH hypothetical_edges (not empty)."""
        candidate = "Test response."
        context = "Test context."

        await self.dreamer.validate_response(
            candidate, context, session_id="test-session-123"
        )

        # Verify diagnose_trace was called with hypothetical_edges
        call_kwargs = mock_sheaf_module.sheaf.diagnose_trace.call_args
        self.assertIn("hypothetical_edges", call_kwargs.kwargs)
        edges = call_kwargs.kwargs["hypothetical_edges"]
        self.assertGreater(len(edges), 0, "hypothetical_edges should not be empty")

    async def test_full_repe_profile_in_metrics(self):
        """TC3: Full RepE profile (all 4 axes) reaches the LLM prompt."""
        candidate = "Test response."
        context = "Test context."

        await self.dreamer.validate_response(candidate, context)

        # The LLM generate call should contain all 4 RepE axes in its prompt
        llm_call_args = self.dreamer.llm.generate.call_args
        prompt = llm_call_args.kwargs.get("prompt", "")
        self.assertIn("Shakiness", prompt)
        self.assertIn("Confluence", prompt)
        self.assertIn("Evasion", prompt)
        self.assertIn("Freedom", prompt)

    async def test_llm_classification_invalid(self):
        """TC4: LLM returns 'invalid' verdict → RLM_DREAMER_ISSUES."""
        candidate = "I assume I am confused."

        # LLM decides this is not grounded
        self.dreamer.llm.generate = AsyncMock(
            return_value=_llm_judgment(
                "invalid",
                0.3,
                reasons=["Low groundedness", "Uncertain language"],
                instruction="Verify with rlm.recall() before re-submitting.",
            )
        )

        result = await self.dreamer.validate_response(candidate, "ctx")

        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["event"], "RLM_DREAMER_ISSUES")
        self.assertIn("Verify", result["instruction"])

    async def test_sheaf_loop_rejection(self):
        """TC5: Topological Loop detected → LLM should reject."""
        candidate = "Looping content."

        mock_sheaf_module.sheaf.diagnose_trace.return_value = {
            "status": "LOGICAL_KNOT",
            "critique": "Loop detected in execution trace.",
            "energy": 2.5,
        }
        self.dreamer.llm.generate = AsyncMock(
            return_value=_llm_judgment(
                "invalid",
                0.2,
                reasons=["Loop in execution"],
                instruction="Break the loop.",
            )
        )

        result = await self.dreamer.validate_response(candidate, "ctx")

        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["event"], "RLM_DREAMER_ISSUES")

    async def test_omcd_forced_stop(self):
        """TC6: oMCD budget exhausted + invalid → escalated SYSTEM CRITICAL."""
        candidate = "Expensive error."

        mock_sheaf_module.sheaf.diagnose_trace.return_value = {
            "status": "SEMANTIC_DRIFT",
            "energy": 1.8,
        }
        mock_omcd_module.omcd.evaluate_step.return_value = {
            "should_stop": True,
            "q_stop": 0.9,
            "threshold": 0.6,
        }
        self.dreamer.llm.generate = AsyncMock(
            return_value=_llm_judgment(
                "invalid", 0.1, reasons=["Drifting"], instruction="Stop."
            )
        )

        result = await self.dreamer.validate_response(candidate, "ctx")

        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["event"], "RLM_DREAMER_ISSUES")
        self.assertIn("SYSTEM CRITICAL", result["instruction"])

    async def test_session_trace_has_timestamps(self):
        """TC7: Session trace includes timestamps and recent_node_ids."""
        trace = self.dreamer._get_session_trace("test-session")

        self.assertIn("recent_node_ids", trace)
        self.assertIn("status_timeline", trace)
        self.assertGreater(len(trace["recent_node_ids"]), 0)
        # Timeline entries should have 'ts' field
        for entry in trace["status_timeline"]:
            self.assertIn("ts", entry)
            self.assertIn("status", entry)


if __name__ == "__main__":
    unittest.main()
