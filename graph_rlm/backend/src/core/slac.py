"""
Core implementation of the SemanticLogicAutoConceptualizer (SLAC) engine and
Arthur Prior's Modal Temporal Logic for auditing past/future consistency in RLM.
"""

import re
from typing import Any, Dict, List

from .logger import get_logger

logger = get_logger("graph_rlm.slac")


class TemporalLogicSystem:
    """
    Implements Arthur Prior's Modal Temporal Logic (Tense Logic) for RLM.
    Used for auditing consistency between past results (H/P) and future plans (G/F).
    """

    @staticmethod
    def get_axioms() -> List[str]:
        """
        Retrieves the core axioms of Prior's Temporal Logic used by the system.

        Returns:
            A list of string representations of the temporal axioms.
        """
        return [
            "FQ ≅ ~G~Q (Future possibility is not always false)",
            "PQ ≅ ~H~Q (Past possibility is not always false)",
            "G(Q -> R) -> (GQ -> GR) (Future distribution)",
            "H(Q -> R) -> (HQ -> HR) (Past distribution)",
            "GQ -> Q (Reflexive future)",
            "HQ -> Q (Reflexive past)",
            "Q -> FPQ (Truth implies future-past perfect)",
            "FFQ -> FQ (Future of future is future)",
        ]

    @staticmethod
    def audit_temporal_consistency(statements: List[str]) -> Dict[str, Any]:
        """
        Analyzes a list of statements for tense consistency and Prior grounding.
        """
        # Patterns for tense detection
        future_pattern = re.compile(
            r"\b(will|shall|going to|future|G|F)\b", re.IGNORECASE
        )
        past_pattern = re.compile(
            r"\b(was|were|had|did|past|P|H|completed|documented|read|fixed|merged|synced|integrated|done|finished|already)\b",
            re.IGNORECASE,
        )

        has_future = any(future_pattern.search(s) for s in statements)
        has_past = any(past_pattern.search(s) for s in statements)

        # Check for contradictions: Planning future (G/F) for something already past (H/P)?
        contradictions = []
        if has_future and has_past:
            full_text = " ".join(statements).lower()
            # If we see future pattern and any completion keywords
            if future_pattern.search(full_text) and any(
                kw in full_text for kw in ["already", "done", "completed", "documented"]
            ):
                # Verify cross-tense inconsistency: future in one, completion in another (or same)
                has_completion = any(
                    (
                        "already" in s.lower()
                        or "done" in s.lower()
                        or "completed" in s.lower()
                    )
                    for s in statements
                )
                if has_completion:
                    contradictions.append(
                        "Cross-Tense Inconsistency: Future plan vs. Completed result."
                    )

        return {
            "tense_span": (
                "Cross-Temporal" if (has_future and has_past) else "Mono-Temporal"
            ),
            "has_future": has_future,
            "has_past": has_past,
            "contradictions": contradictions,
            "status": "INCONSISTENT" if contradictions else "STABLE",
        }


class SLACEngine:
    """
    SemanticLogicAutoConceptualizer (SLAC) Core Engine.
    Formula: A(T) = Truth(T) + alpha * Scrutiny(F) + beta * Improvement(I)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.5):
        self.alpha = alpha
        self.beta = beta
        self.stages = ["C", "R", "F", "S", "D", "RB", "M", "SF"]
        self.stage_names = {
            "C": "Conceptualization",
            "R": "Representation",
            "F": "Facts",
            "S": "Scrutiny",
            "D": "Derivation",
            "RB": "Rule-Based",
            "M": "Model",
            "SF": "Semantic Formalization",
        }

    def calculate_advancement(
        self, truth: float, scrutiny: float, improvement: float
    ) -> float:
        """A(T) = T + aF + bI"""
        return truth + (self.alpha * scrutiny) + (self.beta * improvement)

    def get_stage(self, at_score: float) -> str:
        """Maps Advancement score to SLAC Stage."""
        # Normalize score over stages
        norm = max(0.0, min(1.0, at_score / 3.0))  # A(T) can exceed 1
        idx = int(norm * (len(self.stages) - 1))
        return self.stages[idx]

    def generate_progress_bar(self, stage: str, score: float) -> str:
        """Generates a visual progress bar for the SLAC stage."""
        idx = self.stages.index(stage) if stage in self.stages else 0
        total = len(self.stages)
        progress = (idx + 1) / total

        bar_len = 15
        filled = int(bar_len * progress)
        visual_bar = "█" * filled + "░" * (bar_len - filled)
        return (
            f"[{visual_bar}] {self.stage_names.get(stage, 'Unknown')} (A:{score:.2f})"
        )

    def run_cycle(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a SLAC advancement cycle based on agential metrics.
        metrics Keys: truth (0-1), shakiness (0-1), improvement (delta)
        """
        truth = metrics.get("truth", 0.5)
        # Scrutiny (F): Inverse of shakiness (Low shakiness = high scrutiny of errors)
        scrutiny = 1.0 - metrics.get("shakiness", 0.5)
        improvement = metrics.get("improvement", 0.1)

        at_score = self.calculate_advancement(truth, scrutiny, improvement)
        stage = self.get_stage(at_score)

        return {
            "at_score": at_score,
            "stage": stage,
            "stage_name": self.stage_names.get(stage),
            "progress_bar": self.generate_progress_bar(stage, at_score),
            "status": "ADVANCING" if at_score > 0.7 else "STAGNANT",
        }
