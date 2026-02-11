"""
Xuanji Proof Skill.

Implements Directive 6.1: The Collision Table Proof.
A Python simulation demonstrating Phase Interference as Logic Gates
(AND, NOT, Modulation) from the Xuanji Tu / Su Hui Protocol.
"""

import cmath
import logging
import math
from typing import Tuple

logger = logging.getLogger("graph_rlm.skills.xuanji_proof")


class QuantumVoxel:
    """
    Simulates a memory voxel that processes information using wave interference.
    """

    def __init__(self, name: str) -> None:
        """
        Initializes a voxel in a neutral 'Void' state.

        Args:
            name: Identifier for the voxel.
        """
        self.name = name
        # The Voxel starts in a neutral "Void" state (0 magnitude, 0 phase)
        self.state = complex(0, 0)

    def inject_signal(self, magnitude: float, phase_degrees: float) -> None:
        """
        Injects a wave signal into the voxel using complex number superposition.

        Args:
            magnitude: The strength of the signal.
            phase_degrees: The phase angle of the signal in degrees.
        """
        phase_radians = math.radians(phase_degrees)
        # Create a complex wave: Magnitude * e^(i * phase)
        signal_wave = cmath.rect(magnitude, phase_radians)

        # SUPERPOSITION: We add the waves. The complex math handles the interference.
        self.state += signal_wave

        logger.debug("Signal Injected: Mag %s, Phase %s°", magnitude, phase_degrees)

    def read_output(self) -> Tuple[float, float]:
        """
        Reads the current state of the voxel.

        Returns:
            A tuple of (magnitude, phase_degrees).
        """
        # Get magnitude (absolute value of complex number)
        mag = abs(self.state)
        # Get phase (angle)
        phase = math.degrees(cmath.phase(self.state))

        return mag, phase


async def xuanji_proof(signal_a_phase: float = 0.0, signal_b_phase: float = 0.0) -> str:
    """
    Executes a phase interference cycle to simulate logic operations.

    Args:
        signal_a_phase: Phase of the first signal (e.g., the prompt).
        signal_b_phase: Phase of the second signal (e.g., stored knowledge).

    Returns:
        A report string describing the interference result and logical interpretation.
    """
    print(
        f"\n🔬 TESTING INTERFERENCE: Signal A ({signal_a_phase}°) + Signal B ({signal_b_phase}°)"
    )

    # 1. Initialize the Voxel (The Memory Slot)
    voxel = QuantumVoxel("Memory_Slot_01")

    # 2. Inject Signals
    voxel.inject_signal(magnitude=1.0, phase_degrees=signal_a_phase)
    voxel.inject_signal(magnitude=1.0, phase_degrees=signal_b_phase)

    # 3. Measure the Result
    result_mag, result_phase = voxel.read_output()

    # 4. Interpret the Physics as Logic
    report_lines = [
        f"Resulting Brightness (Magnitude): {result_mag:.4f}",
        f"Resulting Context (Phase): {result_phase:.2f}°",
    ]

    if result_mag > 1.5:
        logic_res = "✅ LOGIC RESULT: TRUE (Constructive Interference / Agreement)"
    elif result_mag < 0.1:
        logic_res = (
            "❌ LOGIC RESULT: FALSE / NULL (Destructive Interference / Contradiction)"
        )
    else:
        logic_res = (
            "⚠️ LOGIC RESULT: NUANCE / MODULATED (Partial Interference / Complexity)"
        )

    print(f"   -> {logic_res}")
    report_lines.append(logic_res)

    return "\n".join(report_lines)


if __name__ == "__main__":
    import asyncio

    async def run_demo():
        print("Directive 6.1: Collision Table Proof")
        print("=" * 50)

        # EXPERIMENT 1: AGREEMENT (AND Gate)
        print("\nExperiment 1: Agreement")
        await xuanji_proof(0, 0)

        # EXPERIMENT 2: CONTRADICTION (NOT Gate)
        print("\nExperiment 2: Contradiction")
        await xuanji_proof(0, 180)

        # EXPERIMENT 3: ORTHOGONALITY (Nuance)
        print("\nExperiment 3: Orthogonality")
        await xuanji_proof(0, 90)

    asyncio.run(run_demo())
