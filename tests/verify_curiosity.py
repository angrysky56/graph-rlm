"""
Verification script for Sheaf-Theoretic Intelligent Curiosity.
Tests the Navigator's ability to compute compression progress and rank actions.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# --- MOCKING DB BEFORE IMPORTS ---
from unittest.mock import MagicMock

sys.modules["graph_rlm.backend.src.core.db"] = MagicMock()
# Also mock the internal db object if accessed directly
mock_db_module = MagicMock()
mock_db_module.db = MagicMock()
sys.modules["graph_rlm.backend.src.core.db"] = mock_db_module

from graph_rlm.backend.src.core.navigator import Navigator
from graph_rlm.backend.src.core.sheaf import SheafMonitor


def test_compression_progress():
    print("\n--- Testing Compression Progress (R(t)) ---")
    nav = Navigator()

    # baseline
    history = [
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the lazy dog again.",
        "Repetition makes things compressible.",
    ]

    for h in history:
        nav.update_history(h)

    print(f"History buffer size: {len(nav.history_buffer)}")

    # Candidate 1: Highly compressible (pattern match)
    candidate_match = "The quick brown fox jumps over the lazy dog three times."
    progress_match = nav.compute_compression_progress(candidate_match)
    print(f"Candidate (Pattern Match): R(t) = {progress_match:.4f}")

    # Candidate 2: Random noise (incompressible)
    candidate_noise = "xcvbnm,./1234567890-="
    progress_noise = nav.compute_compression_progress(candidate_noise)
    print(f"Candidate (Noise): R(t) = {progress_noise:.4f}")

    if progress_match > progress_noise:
        print(
            "✅ SUCCESS: Pattern match yields higher compression progress than noise."
        )
    else:
        print("❌ FAILURE: Compression progress metric is not working as expected.")


def test_future_entropy():
    print("\n--- Testing Future Entropy (S_tau) ---")
    nav = Navigator()

    # Candidate 1: High Entropy (Exploration)
    cand_explore = "I will list all files in the directory to explore structure."
    entropy_explore = nav.estimate_future_entropy(cand_explore)
    print(f"Candidate (Explore): S_tau = {entropy_explore:.4f}")

    # Candidate 2: Low Entropy (Restriction)
    cand_restrict = "I will delete the file and assert that it is gone."
    entropy_restrict = nav.estimate_future_entropy(cand_restrict)
    print(f"Candidate (Restrict): S_tau = {entropy_restrict:.4f}")

    if entropy_explore > entropy_restrict:
        print("✅ SUCCESS: Exploration action has higher future entropy.")
    else:
        print("❌ FAILURE: Entropy estimation is not working as expected.")


def test_sheaf_integration():
    print("\n--- Testing Sheaf Integration (Laplacian) ---")

    monitor = SheafMonitor()

    # Construct a simple graph
    nodes = [{"id": "A"}, {"id": "B"}, {"id": "C"}]
    edges = [("A", "B"), ("B", "C"), ("C", "A")]  # Cycle

    try:
        L = monitor.compute_sheaf_laplacian(nodes, edges)
        print("Sheaf Laplacian shape:", L.shape)
        print(L)
        print("✅ SUCCESS: Sheaf Laplacian computed.")
    except Exception as e:
        print(f"❌ FAILURE: Sheaf Laplacian computation failed: {e}")


def main():
    test_compression_progress()
    test_future_entropy()
    test_sheaf_integration()


if __name__ == "__main__":
    main()
