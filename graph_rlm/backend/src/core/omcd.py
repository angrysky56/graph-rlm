"""
oMCD (online Metacognitive Control of Decisions) Controller.

Implements optimal resource allocation and stopping decisions for the Agent.

Formulas:
- Benefit:  B(z) = R * P_c(z)
- Cost:     C(z) = α * z^ν
- Optimal Stopping: Q(a=STOP, t) = R * P_c(t) - α * (κ * t)^ν
- Precision Update: 1/σ(z) = 1/σ_0 + β * z
"""

from dataclasses import dataclass
from typing import Optional

from .logger import get_logger
from .trace import trace_action

logger = get_logger("graph_rlm.omcd")


@dataclass
class OmcdParams:
    """Tunable parameters for the oMCD model."""

    # Importance weight for confident decisions (default: 1.0 = important)
    r: float = 1.0

    # Unitary effort cost (higher = more expensive per step)
    alpha: float = 0.01

    # Cost power (ν > 1 = cost accelerates with steps)
    nu: float = 1.5

    # Stopping threshold (must exceed this to commit)
    omega: float = 0.7

    # Effort intensity (resources per unit time, default: 1)
    kappa: float = 1.0

    # Precision efficacy (sharpens confidence with effort)
    beta: float = 0.1

    # Value mode exploration noise scale
    gamma: float = 0.05

    # Minimum steps before oMCD can recommend stopping.
    # Prevents premature termination on the first few steps.
    min_steps: int = 3


class OmcdController:
    """
    Metacognitive Controller for Optimal Decision Making.

    Usage:
        controller = OmcdController()
        for step in range(max_steps):
            confidence = sheaf.diagnose_trace(...)["confidence"]
            decision = controller.evaluate_step(step, confidence)
            if decision["should_stop"]:
                break
    """

    def __init__(self, params: Optional[OmcdParams] = None):
        self.params = params or OmcdParams()
        self._history: list[dict] = []
        self._last_bernshteyn_score: float = 0.0

    def calculate_bernshteyn_penalty(self, depth: int, precision: float) -> float:
        """
        Calculates the Bernshteyn Penalty (Rulial Singularity detector).
        Based on the bound: p(d+1)^8 <= 2^-15.

        If the bound is violated, the cost of deliberation increases exponentially.
        """
        # p = precision (0..1), d = depth
        # We check if p * (d + 1)^8 > 2^-15
        threshold = 2**-15

        # Ensure precision is at least a small epsilon to avoid zeroing out too early
        p = max(1e-10, precision)
        val = p * ((depth + 1) ** 8)

        self._last_bernshteyn_score = val

        if val > threshold:
            # Singularity detected: increase cost multiplier
            import math

            # Using k * log(val/threshold) to create a steep but stable penalty
            # k=10 makes it quite aggressive
            penalty = 1.0 + 10.0 * math.log(val / threshold)

            trace_action(
                "oMCD",
                "BERNSHTEYN_LIMIT_BREACH",
                result=f"Singularity Breach: {val:.2e} > {threshold:.2e}. Penalty: {penalty:.2f}",
                tag="SYSTEM",
            )
            return max(1.0, penalty)
        return 1.0

    def calculate_cost(self, z: float) -> float:
        """
        Calculate the cost of cognitive resources invested.

        C(z) = α * z^ν
        """
        return self.params.alpha * (z**self.params.nu)

    def calculate_benefit(self, confidence: float) -> float:
        """
        Calculate the benefit of the current decision.

        B(z) = R * P_c(z)
        """
        return self.params.r * confidence

    def calculate_q_stop(self, step: int, confidence: float) -> float:
        """
        Calculate the Q-value for stopping (committing to current answer).

        Q(a=STOP, t) = R * P_c(t) - α * (κ * t)^ν
        """
        benefit = self.calculate_benefit(confidence)
        cost = self.calculate_cost(step * self.params.kappa)
        return benefit - cost

    def evaluate_step(
        self, step: int, confidence: float, potential_energy: float = 1.0
    ) -> dict:
        """
        Evaluate whether to continue deliberating or stop.
        Enforces Hamiltonian Energy Conservation to reject physical hallucinations.

        Returns:
            dict containing decision parameters including `is_non_physical`.
        """
        # 1. Calculate Bernshteyn Penalty (Cost of Rulial Noise)
        penalty = self.calculate_bernshteyn_penalty(step, 1.0 - confidence)

        # 2. Physics Engine: Hamiltonian Dynamics
        # T (Kinetic Energy) = physical computational cost exerted
        base_kinetic = self.calculate_cost(step * self.params.kappa)
        kinetic_energy = base_kinetic * penalty

        # V (Potential Energy) = remaining semantic distance to goal
        # Total Energy H = T + V
        hamiltonian = kinetic_energy + potential_energy

        is_non_physical = False

        if self._history:
            last_v = self._history[-1].get("potential_energy", potential_energy)
            progress = last_v - potential_energy  # Postive if we got closer to goal

            # Energy conservation bound (Free Lunch Theorem).
            # The progress made (-Delta V) cannot exceed the base kinetic effort applied.
            # If the agent claims massive progress with minimal base effort, it's an unearned leap (hallucination).
            if progress > 0.5 and base_kinetic < 0.2:
                is_non_physical = True
                trace_action(
                    "oMCD",
                    "CONSERVATION_VIOLATION",
                    result=f"Unearned Leap: Progress={progress:.2f}, Effort={kinetic_energy:.2f}. Hallucination blocked.",
                    tag="SYSTEM",
                )

        benefit = self.calculate_benefit(confidence)

        # Q_stop = Benefit - Cost
        q_stop = benefit - kinetic_energy

        decision = {
            "should_stop": (
                q_stop >= self.params.omega
                and not is_non_physical
                and step >= self.params.min_steps
            ),
            "is_non_physical": is_non_physical,
            "q_stop": q_stop,
            "threshold": self.params.omega,
            "cost": kinetic_energy,
            "benefit": benefit,
            "confidence": confidence,
            "step": step,
            "bernshteyn_penalty": penalty,
            "hamiltonian": hamiltonian,
            "potential_energy": potential_energy,
            "kinetic_energy": kinetic_energy,
            "rationale": (
                f"Stop={q_stop >= self.params.omega}. q_stop={q_stop:.3f} (Benefit={benefit:.2f}, Cost={kinetic_energy:.2f}). "
                f"Bernshteyn={penalty:.2f}. "
                + (
                    "Hallucination detected (Kinetic < Progress)."
                    if is_non_physical
                    else "Physics consistent."
                )
            ),
        }

        # Record for calibration
        self._history.append(decision)

        # Emit trace for visibility
        status_msg = (
            f"Q_stop={q_stop:.3f} (Conf={confidence:.2f}, Cost={kinetic_energy:.3f})"
        )
        if decision["should_stop"]:
            trace_action(
                "oMCD",
                "OPTIMAL_STOP",
                result=f"{status_msg} >= ω={self.params.omega:.2f}. Committing.",
                tag="SYSTEM",
            )
        else:
            # Mirror to terminal for user visibility of deliberation progress
            logger.info(
                "[oMCD] Deliberating... %s < ω=%.2f",
                status_msg,
                self.params.omega,
            )

        return decision

    def update_precision(self, initial_sigma: float, z: int) -> float:
        """
        Update precision of value representation with effort.

        1/σ(z) = 1/σ_0 + β * z
        Returns new sigma (smaller = more precise).
        """
        if initial_sigma <= 0:
            return initial_sigma
        new_precision = (1 / initial_sigma) + (self.params.beta * z)
        return 1 / new_precision if new_precision > 0 else initial_sigma

    def calibrate_from_session(
        self, session_surprise_avg: float, session_improvement_rate: float
    ):
        """
        SLAP-aligned calibration: Adjust α and β based on session outcomes.

        Per SLAP: α + β = 1 (normalized weights).
        """
        if session_surprise_avg + session_improvement_rate > 0:
            total = session_surprise_avg + session_improvement_rate
            alpha_learned = session_surprise_avg / total
            # Map SLAP β to oMCD alpha (cost weight)
            self.params.alpha = alpha_learned * 0.1  # Scale to sensible range
            logger.info(
                "[oMCD] Calibrated: α=%.4f (from surprise %.2f, improvement %.2f)",
                self.params.alpha,
                session_surprise_avg,
                session_improvement_rate,
            )
        trace_action(
            "oMCD",
            "CALIBRATED",
            result=f"α={self.params.alpha:.4f}, ν={self.params.nu:.2f}, ω={self.params.omega:.2f}",
            tag="SYSTEM",
        )

    def get_history(self) -> list[dict]:
        """Return the decision history for debugging/analysis."""
        return self._history

    def reset_history(self):
        """Clear the decision history for a new session."""
        self._history = []


# Singleton instance
omcd = OmcdController()
