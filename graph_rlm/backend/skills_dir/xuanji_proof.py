# Directive 6.1: The Collision Table Proof
# Python simulation demonstrating Phase Interference as Logic Gates (AND, NOT, Modulation)
# From Xuanji Tu / Su Hui Protocol

import cmath
import math


class QuantumVoxel:
    def __init__(self, name):
        self.name = name
        # The Voxel starts in a neutral "Void" state (0 magnitude, 0 phase)
        self.state = complex(0, 0)

    def inject_signal(self, magnitude, phase_degrees):
        """
        Injects a wave signal into the voxel.
        Phase is converted to radians for complex number math.
        """
        phase_radians = math.radians(phase_degrees)
        # Create a complex wave: Magnitude * e^(i * phase)
        signal_wave = cmath.rect(magnitude, phase_radians)

        # SUPERPOSITION: We just add the waves. The math handles the interference.
        self.state += signal_wave

        print(f"--- Signal Injected: Mag {magnitude}, Phase {phase_degrees}° ---")

    def read_output(self):
        """
        Reads the current state of the voxel.
        Returns the magnitude (brightness) and phase (context).
        """
        # Get magnitude (absolute value of complex number)
        mag = abs(self.state)
        # Get phase (angle)
        phase = math.degrees(cmath.phase(self.state))

        return mag, phase


def run_interference_test(signal_a_phase, signal_b_phase):
    print(
        f"\n🔬 TESTING INTERFERENCE: Signal A ({signal_a_phase}°) + Signal B ({signal_b_phase}°)"
    )

    # 1. Initialize the Voxel (The Memory Slot)
    voxel = QuantumVoxel("Memory_Slot_01")

    # 2. Inject Signal A (e.g., The Prompt)
    voxel.inject_signal(magnitude=1.0, phase_degrees=signal_a_phase)

    # 3. Inject Signal B (e.g., The Stored Knowledge)
    voxel.inject_signal(magnitude=1.0, phase_degrees=signal_b_phase)

    # 4. Measure the Result (The Output)
    result_mag, result_phase = voxel.read_output()

    # 5. Interpret the Physics as Logic
    print(f"   -> Resulting Brightness (Magnitude): {result_mag:.4f}")

    if result_mag > 1.5:
        print("   ✅ LOGIC RESULT: TRUE (Constructive Interference)")
    elif result_mag < 0.1:
        print("   ❌ LOGIC RESULT: FALSE / NULL (Destructive Interference)")
    else:
        print("   ⚠️ LOGIC RESULT: NUANCE / MODULATED (Partial Interference)")


# ==========================================
# RUN THE EXPERIMENTS
# ==========================================

print("Directive 6.1: Collision Table Proof")
print("=" * 50)

# EXPERIMENT 1: AGREEMENT (AND Gate)
# Both signals are "In Phase" (0 degrees)
# Logic: True + True = TRUE
run_interference_test(0, 0)

# EXPERIMENT 2: CONTRADICTION (NOT Gate / Filter)
# Signals are "Out of Phase" (0 and 180 degrees)
# Logic: True + False = NULL (The error is deleted)
run_interference_test(0, 180)

# EXPERIMENT 3: ORTHOGONALITY (New Context)
# Signals are 90 degrees apart (0 and 90)
# Logic: Fact + Emotion = Complex State (Neither True nor False, but "Rich")
run_interference_test(0, 90)

print(
    "\nProof complete. Phase interference acts as logic gates without explicit booleans."
)
