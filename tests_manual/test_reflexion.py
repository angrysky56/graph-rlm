
import asyncio
import numpy as np
import sys
import os

# Ensure the package is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph_rlm.backend.src.core.reflexion import intelli_synth

async def test_reflexion():
    print("Testing IntelliSynth Framework...")

    # 1. Math Functions
    print(f"Sigmoid(0): {intelli_synth.sigmoid(0)}")
    print(f"Entropy([0.5, 0.5]): {intelli_synth.entropy([0.5, 0.5])}")
    print(f"Bayes Reason(0.1, 0.5, 0.05): {intelli_synth.reason(0.1, 0.5, 0.05)}")

    # 2. AwL Suite
    logic = await intelli_synth.analyze_with_logic(["A -> B", "A is true"])
    print(f"Logic Result: {logic}")

    intuition = await intelli_synth.engage_intuition(["Repetitive failure in trace"])
    print(f"Intuition Score: {intuition}")

    abduction, plaus = await intelli_synth.employ_abductive_reasoning(["Trace shows empty result"])
    print(f"Abduction assumptions: {abduction}")

    # 3. Advancement Cycle (Mock prompt)
    # Mocking RepE and Sheaf to avoid DB connections
    from unittest.mock import MagicMock
    import sys

    mock_repe = MagicMock()
    mock_repe.scan_thought.return_value = {"Shakiness": 0.1}

    mock_sheaf = MagicMock()
    mock_sheaf.diagnose_trace.return_value = {"status": "HEALTHY", "energy": 0.1}

    # Patch modules
    # We need to patch where they are imported FROM, which is .repe and .sheaf in reflexion.py
    # But since they are lazy imported, we can inject them into sys.modules
    sys.modules["graph_rlm.backend.src.core.repe"] = MagicMock(repe=mock_repe)
    sys.modules["graph_rlm.backend.src.core.sheaf"] = MagicMock(sheaf=mock_sheaf)

    directive = await intelli_synth.advancement_cycle(
        trace_context="Agent keeps trying 'ls' on a non-existent directory.",
        current_thought="I am trying to find the directory.",
        divergence_point="File listing failed 3 times"
    )
    print(f"Improvement Directive: {directive}")

    print("\n✅ IntelliSynth Verification Complete.")

if __name__ == "__main__":
    asyncio.run(test_reflexion())
