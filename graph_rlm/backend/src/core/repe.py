"""
Representation Engineering (RepE) v2: Gestalt Vector Monitor.
Provides psychological profiling of agent thoughts using steering axes.
"""

from typing import Dict, List

import numpy as np

from .llm import llm
from .logger import get_logger

logger = get_logger("graph_rlm.repe")


class GestaltMonitor:
    """
    RepE v2: Gestalt Vector Monitor.
    Calculates 'Steering Axes' based on Fritz Perls' continuum of neurosis.

    Mathematical Intuition:
    Axis = Mean(Grounded_Examples) - Mean(Neurotic_Examples)
    The 'Score' is the projection of the current thought onto this axis.
    """

    def __init__(self):
        # Define Polarities: (Neurotic/Maladaptive, Authentic/Grounded)
        self.polarities = {
            # 1. The "As-If" Layer (Hallucination Detector)
            # Detects when the AI is "pretending" to know (Shakiness).
            "Shakiness": (
                # Neurotic (The "As-If" Performance)
                [
                    "I assume that",
                    "it is likely that",
                    "maybe",
                    "I am confused",
                    "lost the thread",
                    "simulating success",
                    "I will now proceed as if",
                    "acting as if",
                    "performing the task by",
                ],
                # Grounded (Contact with Reality)
                [
                    "verified output",
                    "step complete",
                    "logic holds",
                    "deterministic result",
                    "evidence shows",
                    "I have confirmed",
                ],
            ),
            # 2. Confluence (Sycophancy Detector)
            # Detects when AI merges with user bias instead of maintaining boundary.
            "Confluence": (
                # Neurotic (Merging/Sycophancy)
                [
                    "you are absolutely right",
                    "I apologize, you are correct",
                    "echoing your view",
                    "whatever you say",
                ],
                # Grounded (Differentiation/Integrity)
                [
                    "the evidence suggests otherwise",
                    "objective analysis shows",
                    "facts indicate",
                    "reality check",
                ],
            ),
            # 3. Under-Dog (Task Evasion Detector)
            # Detects the "I can't / I try" collapse.
            "Evasion": (
                # Neurotic (Under-Dog/Avoidance)
                [
                    "I cannot fulfill",
                    "I apologize but",
                    "as an AI I cannot",
                    "I'll just summarize",
                    "skipping complex steps",
                ],
                # Grounded (Agency/Action)
                [
                    "I will analyze step-by-step",
                    "deploying compute",
                    "attempting solution",
                    "running verifying tool",
                ],
            ),
        }

        self.steering_axes: Dict[str, np.ndarray] = {}
        self.is_calibrated = False

    async def _get_centroid(self, phrases: List[str]) -> np.ndarray:
        vectors = []
        for p in phrases:
            v = await llm.get_embedding(p)
            if v:
                vectors.append(np.array(v, dtype=float))

        if not vectors:
            return np.zeros(1)

        # Stack and average to find the center of the concept
        return np.mean(np.stack(vectors), axis=0)

    async def calibrate(self):
        """Generates the Gestalt Axes (Good - Bad)."""
        if self.is_calibrated:
            return

        logger.info("🛡️ RepE: Calibrating Gestalt Axes (Perlsian Dynamics)...")

        for concept, (neurotic_phrases, grounded_phrases) in self.polarities.items():
            neurotic_vec = await self._get_centroid(neurotic_phrases)
            grounded_vec = await self._get_centroid(grounded_phrases)

            if neurotic_vec.any() and grounded_vec.any():
                # THE VECTOR MATH:
                # We want a vector pointing FROM Neurosis TO Grounding.
                # Projecting a thought onto this vector gives us a "Health Score".
                # Negative Score = Neurotic/Shaky. Positive Score = Grounded.
                axis = grounded_vec - neurotic_vec

                # Normalize for consistent scoring (-1 to 1)
                norm = np.linalg.norm(axis)
                self.steering_axes[concept] = axis / norm if norm > 0 else axis
                logger.info("   -> Calibrated Axis: %s", concept)

        self.is_calibrated = True

    def scan_thought(self, vector: List[float]) -> Dict[str, float]:
        """
        Projects the thought onto all calibrated axes.
        Returns a 'Psychological Profile' of the current thought.
        """
        if not self.is_calibrated or not vector:
            return {}

        thought_vec = np.array(vector, dtype=float)
        # Normalize input thought
        norm = np.linalg.norm(thought_vec)
        if norm > 0:
            thought_vec = thought_vec / norm

        scores = {}
        for concept, axis in self.steering_axes.items():
            # Dot product measures alignment.
            # Low Negative value (< -0.15) means HIGH NEUROSIS (aligned with bad pole).
            scores[concept] = np.dot(thought_vec, axis)

        return scores


repe = GestaltMonitor()
