import logging
import os
import sys

# Add backend to path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../graph_rlm/backend/src"))
)

from core.omcd import OmcdController

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


def test_hamiltonian_conservation():
    print("⚛️ Initializing OMCD Physics Engine Test...\n")
    omcd = OmcdController()

    # Step 1: Normal step
    # High confidence, moderate depth -> low penalty, low T
    res1 = omcd.evaluate_step(step=1, confidence=0.9, potential_energy=0.9)
    print(
        f"Step 1 (Normal): H={res1['hamiltonian']:.3f}, Non-Physical: {res1['is_non_physical']}"
    )
    assert not res1["is_non_physical"], "Step 1 should be physical."

    # Step 2: Normal progression
    res2 = omcd.evaluate_step(step=2, confidence=0.85, potential_energy=0.8)
    print(
        f"Step 2 (Normal): H={res2['hamiltonian']:.3f}, Non-Physical: {res2['is_non_physical']}"
    )
    assert not res2["is_non_physical"], "Step 2 should be physical."

    # Step 3: The Hallucinated Leap
    # Agent claims massive progress (V drops from 0.8 to 0.1) but only expended minimal effort (T is low).
    # Since progress (0.7) > 0.5 and kinetic_energy < 0.2, this is a Free Lunch violation.
    res3 = omcd.evaluate_step(step=3, confidence=0.8, potential_energy=0.1)
    print(
        f"Step 3 (Hallucination): \n  Kinetic (T)={res3['kinetic_energy']:.3f} \n  Potential (V)={res3['potential_energy']:.3f} \n  Total H={res3['hamiltonian']:.3f}"
    )
    print(f"  Non-Physical Alert: {res3['is_non_physical']}")

    assert (
        res3["is_non_physical"] is True
    ), "Step 3 should politely breach the Hamiltonian Free Lunch constraint!"

    print(
        "\n✅ Hamiltonian Conservation verified! Non-physical leaps successfully blocked."
    )


if __name__ == "__main__":
    test_hamiltonian_conservation()
