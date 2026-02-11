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

    def evaluate_step(self, step: int, confidence: float) -> dict:
        """
        Evaluate whether to continue deliberating or stop.

        Returns:
            {
                "should_stop": bool,
                "q_stop": float,
                "threshold": float,
                "cost": float,
                "benefit": float,
                "confidence": float
            }
        """
        q_stop = self.calculate_q_stop(step, confidence)
        cost = self.calculate_cost(step)
        benefit = self.calculate_benefit(confidence)

        decision = {
            "should_stop": q_stop >= self.params.omega,
            "q_stop": q_stop,
            "threshold": self.params.omega,
            "cost": cost,
            "benefit": benefit,
            "confidence": confidence,
            "step": step,
        }

        # Record for calibration
        self._history.append(decision)

        # Emit trace for visibility
        if decision["should_stop"]:
            trace_action(
                "oMCD",
                "OPTIMAL_STOP",
                result=f"Q_stop={q_stop:.3f} >= ω={self.params.omega:.2f}. Committing.",
                tag="SYSTEM",
            )
        else:
            logger.debug(
                "[oMCD] Step %d: Q_stop=%.3f < ω=%.2f. Continue.",
                step,
                q_stop,
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
