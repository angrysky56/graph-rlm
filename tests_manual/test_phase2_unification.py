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

def test_phase2_unification():
    print("⚛️ Testing Phase 2 Hamiltonian Unification (Goal Dist + Sheaf Energy)...\n")
    omcd = OmcdController()

    # Scenario: High Goal Distance, High Inconsistency
    # Goal Dist = 0.8, Sheaf Energy = 0.5 -> V = 1.3
    res1 = omcd.evaluate_step(step=1, confidence=0.5, potential_energy=1.3)
    print(f"Step 1 (Stressed): V={res1['potential_energy']:.3f}, H={res1['hamiltonian']:.3f}")

    # Scenario: Agent claims to reach goal but is MODERATELY inconsistent
    # Goal Dist = 0.1 (claims success), Sheaf Energy = 0.6 (topological defect)
    # V = 0.1 + 0.6 = 0.7
    # Progress from Step 1: 1.3 - 0.7 = 0.6
    # If effort (T) is low (step 2 -> base_kinetic ~ 0.05 * 2^1.5 ~ 0.14),
    # then progress 0.6 > 0.5 AND effort < 0.2 -> NON_PHYSICAL!
    res2 = omcd.evaluate_step(step=2, confidence=0.9, potential_energy=0.7)
    print(f"Step 2 (Consistent Goal but Inconsistent Logic):")
    print(f"  V={res2['potential_energy']:.3f}, T={res2['kinetic_energy']:.3f}, H={res2['hamiltonian']:.3f}")
    print(f"  Non-Physical Alert: {res2['is_non_physical']}")

    assert res2["is_non_physical"] is True, "Moderately inconsistent logic should trigger non-physical leap if goal distance dropped sharply with low effort."

    # Scenario: Agent reaches goal and IS consistent
    # Reset history to simulate a clean start for the success case
    omcd.reset_history()

    # Goal Dist = 0.1, Sheaf Energy = 0.05 -> V = 0.15
    # Use very high confidence (0.9999999) to avoid Bernshteyn penalty for d=1
    res3 = omcd.evaluate_step(step=1, confidence=0.9999999, potential_energy=0.15)
    print(f"\nStep 1 (Consistent Goal and Logic):")
    print(f"  V={res3['potential_energy']:.3f}, T={res3['kinetic_energy']:.3f}, H={res3['hamiltonian']:.3f}")
    print(f"  Should Stop: {res3['should_stop']}")

    assert not res3["is_non_physical"], "Step 1 should be physical."
    assert res3["should_stop"], "Agent should stop when both goal and logic are consistent and cost is low."

    print("\n✅ Phase 2 Hamiltonian Unification verified!")

if __name__ == "__main__":
    test_phase2_unification()
