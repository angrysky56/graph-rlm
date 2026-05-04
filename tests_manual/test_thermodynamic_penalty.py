import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../graph_rlm/backend/src")))

from core.navigator import Navigator


async def test_thermodynamic_penalty():
    print("🌡️ Testing Thermodynamic Penalty (Cost of Determinism)...")
    navigator = Navigator()

    # Mock compression progress (r_t) to be constant so we only test the entropy effect
    navigator.compute_compression_progress = MagicMock(return_value=0.5)

    candidates = ["Path A (High Freedom)", "Path B (Low Freedom, Deterministic)"]

    # We mock estimate_future_entropy to simulate RepE responses.
    # High entropy (Freedom score ~ 0.8) -> Normalized: (0.8+1)/2 = 0.90
    # Low entropy (Freedom score ~ -0.8) -> Normalized: (-0.8+1)/2 = 0.10
    async def mock_entropy(cand):
        if "High Freedom" in cand:
            return 0.90
        elif "Low Freedom" in cand:
            return 0.10
        return 0.50

    navigator.estimate_future_entropy = AsyncMock(side_effect=mock_entropy)

    ranked = await navigator.compute_interest_gradient(candidates, history=[])

    # ranked is a list of tuples: (score, details)
    # details dict contains our new thermodynamic_penalty metric
    results = {details["content"]: details for score, details in ranked}

    high_freedom = results["Path A (High Freedom)"]
    low_freedom = results["Path B (Low Freedom, Deterministic)"]

    print(f"\nPath A | Entropy: {high_freedom['future_entropy']:.2f} | Penalty: {high_freedom['thermodynamic_penalty']:.2f} | Score: {high_freedom['score']:.4f}")
    print(f"Path B | Entropy: {low_freedom['future_entropy']:.2f} | Penalty: {low_freedom['thermodynamic_penalty']:.2f} | Score: {low_freedom['score']:.4f}")

    # Assertions
    assert high_freedom["thermodynamic_penalty"] == 0.0, "High freedom path should have NO penalty."
    assert low_freedom["thermodynamic_penalty"] > 0.0, "Low freedom path MUST incur a thermodynamic penalty."
    assert high_freedom["score"] > low_freedom["score"], "High freedom should score higher due to penalty on the deterministic path."

    print("\n✅ Thermodynamic Penalty working correctly.")

if __name__ == "__main__":
    asyncio.run(test_thermodynamic_penalty())
