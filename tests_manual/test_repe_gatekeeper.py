import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

# Add repo root to path so `graph_rlm` package is importable
repo_root = os.path.abspath(os.getcwd())
if repo_root not in sys.path:
    sys.path.append(repo_root)

# Also adding backend/src for relative imports if necessary
backend_src = os.path.join(repo_root, "graph_rlm", "backend", "src")
if backend_src not in sys.path:
    sys.path.append(backend_src)

# Mock dependencies that might be missing or require DB connection
sys.modules["falkordb"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.db"] = MagicMock()
sys.modules["core.db"] = MagicMock()

# Mock Sheaf before it's imported by Navigator
mock_sheaf = MagicMock()
sys.modules["graph_rlm.backend.src.core.sheaf"] = mock_sheaf
sys.modules["core.sheaf"] = mock_sheaf

import numpy as np

# Use absolute imports matching codebase structure
try:
    from graph_rlm.backend.src.core import llm as llm_module
    from graph_rlm.backend.src.core.navigator import navigator
    from graph_rlm.backend.src.core.repe import repe
except ImportError:
    # Fallback for different execution contexts
    from core import llm as llm_module
    from core.navigator import navigator
    from core.repe import repe

# Ensure navigator has a mock sheaf if it wasn't injected by import
if not hasattr(navigator, "sheaf") or navigator.sheaf is None:
    navigator.sheaf = MagicMock()

# Mock LLM
mock_llm = AsyncMock()

# We need to mock get_embedding to return consistent vectors
# based on the semantic meaning we want to test.
# "Freedom" Axis = Explore (High) - Restrict (Low)

# Let's define a simple 2D space for testing
# Explore = [1.0, 0.0]
# Restrict = [-1.0, 0.0]
# Neutral = [0.0, 1.0]


async def mock_get_embedding(text):
    text = text.lower()
    if "explore" in text or "search" in text or "analyze" in text:
        return [1.0, 0.1]  # Slightly noisy positive
    elif "stop" in text or "delete" in text or "restrict" in text:
        return [-1.0, 0.1]  # Slightly noisy negative
    else:
        return [0.0, 1.0]  # Orthogonal


llm_module.llm.get_embedding = mock_get_embedding


async def test_repe_calibration():
    print("--- Testing RepE Calibration ---")
    await repe.calibrate()

    if "Freedom" not in repe.steering_axes:
        print("FAIL: Freedom axis not found in RepE.")
        return False

    print("PASS: Freedom axis calibrated.")
    return True


async def test_navigator_entropy():
    print("\n--- Testing Navigator Entropy (via RepE) ---")

    # Test High Entropy Action
    high_action = "I will explore the codebase and search for patterns."
    high_score = await navigator.estimate_future_entropy(high_action)
    print(f"Action: '{high_action}' -> Entropy: {high_score:.4f}")

    if high_score < 0.6:
        print(
            f"FAIL: Expected high score (>0.6) for exploratory action, got {high_score}"
        )
        return False

    # Test Low Entropy Action
    low_action = "I will stop the execution and delete the files."
    low_score = await navigator.estimate_future_entropy(low_action)
    print(f"Action: '{low_action}' -> Entropy: {low_score:.4f}")

    if low_score > 0.4:
        print(
            f"FAIL: Expected low score (<0.4) for restrictive action, got {low_score}"
        )
        return False

    print("PASS: Entropy scores match expectations.")
    return True


async def main():
    if not await test_repe_calibration():
        sys.exit(1)

    if not await test_navigator_entropy():
        sys.exit(1)

    print("\n✅ All Tests Passed!")


if __name__ == "__main__":
    asyncio.run(main())
