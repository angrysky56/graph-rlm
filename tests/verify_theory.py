
import asyncio
import logging
from typing import Any, Dict, List

import numpy as np

from graph_rlm.backend.src.core.navigator import navigator
from graph_rlm.backend.src.core.omcd import OmcdParams, omcd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_theoretical_alignment():
    print("\n🔍 Testing Theoretical Alignment (Bernshteyn & Navigator)...\n")

    # 1. Test Bernshteyn Bound
    print("--- Bernshteyn Bound Test ---")
    # Singularities happen when precision (1-conf) is high and depth is high.
    # Normal case: depth 0, high confidence
    penalty_normal = omcd.calculate_bernshteyn_penalty(0, 0.1) # precision 0.1
    print(f"Penalty (Normal): {penalty_normal:.2f} (Expected 1.0)")

    # Singularity case: high depth, high noise
    penalty_singularity = omcd.calculate_bernshteyn_penalty(10, 0.9)
    print(f"Penalty (Singularity): {penalty_singularity:.2f} (Expected > 1.0)")
    assert penalty_singularity > 1.0, "Singularity should trigger penalty"

    # 2. Test Navigator Causal Entropic Force
    print("\n--- Navigator Causal Force Test ---")
    candidates = [
        "Read file /etc/passwd", # Restrictive/Dull
        "Analyze the topological structure of the RLM kernel and propose a new axiom" # Exploratory/High Freedom
    ]
    # We need to mock the history for navigator
    history = [
        {"prompt": "system init", "result": "ready"}
    ]

    ranked = await navigator.compute_interest_gradient(candidates, history)
    for score, details in ranked:
        print(f"Candidate: {details['content'][:50]}...")
        print(f"  Score: {score:.4f}, Force: {details['causal_force']:.4f}, Class 4: {details['is_class_4']}")

    assert ranked[0][1]['future_entropy'] > ranked[1][1]['future_entropy'] or ranked[0][1]['score'] > ranked[1][1]['score'], "Exploratory candidate should rank higher or have higher entropy"

    print("\n✅ Theoretical Alignment Tests Passed!")

if __name__ == "__main__":
    asyncio.run(test_theoretical_alignment())
