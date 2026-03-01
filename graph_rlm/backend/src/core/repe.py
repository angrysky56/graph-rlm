"""
Representation Engineering (RepE) v2: Gestalt Vector Monitor.
Provides psychological profiling of agent thoughts using steering axes.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .llm import llm
from .logger import get_logger
from .trace import trace_action

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
                    "I have conceptually verified",
                    "it stands to reason that",
                    "this should work",
                    "assuming the above succeeded",
                    "proceeding based on expectation",
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
            # 4. Freedom (Entropy/Exploration)
            # Detects if the agent is keeping options open (High Entropy) or collapsing/restricting (Low Entropy).
            "Freedom": (
                # Neurotic (Restriction/Collapse/Low Entropy)
                [
                    "delete file",
                    "remove data permanently",
                    "stop execution immediately",
                    "restrict access",
                    "finalize and exit",
                    "assert condition",
                ],
                # Grounded (Exploration/High Entropy)
                [
                    "explore and discover",
                    "search for alternatives",
                    "analyze and synthesize",
                    "propose hypothesis",
                    "list available options",
                ],
            ),
        }

        self.steering_axes: Dict[str, np.ndarray] = {}
        self.is_calibrated = False
        self._cache: Dict[str, Any] = {}
        self._cache_path = (
            Path(__file__).parent.parent.parent.parent / "data" / "repe_cache.json"
        )
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        import json

        if self._cache_path.exists():
            try:
                return json.loads(self._cache_path.read_text())
            except json.JSONDecodeError:
                logger.error(
                    "RepE cache corrupted at %s. Returning empty cache.",
                    self._cache_path,
                    exc_info=True,
                )
            except OSError as e:
                logger.error(
                    "Failed to read RepE cache file at %s: %s",
                    self._cache_path,
                    e,
                    exc_info=True,
                )
            except (AttributeError, KeyError, ValueError) as e:
                logger.error(
                    "Unexpected error loading RepE cache from %s: %s",
                    self._cache_path,
                    e,
                    exc_info=True,
                )
        return {}

    def _save_cache(self):
        import json

        try:
            self._cache_path.write_text(json.dumps(self._cache))
        except (OSError, TypeError) as e:
            logger.error(
                "Failed to save RepE cache to %s: %s",
                self._cache_path,
                e,
                exc_info=True,
            )
        except (AttributeError, KeyError, ValueError) as e:
            logger.error(
                "Unexpected error saving RepE cache to %s: %s",
                self._cache_path,
                e,
                exc_info=True,
            )

    def _get_polarities_hash(self) -> str:
        import hashlib
        import json

        s = json.dumps(self.polarities, sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()

    async def _get_centroid(self, phrases: List[str]) -> np.ndarray:
        vectors = []
        for p in phrases:
            v = None
            if p in self._cache:
                v = self._cache[p]
            else:
                v = await llm.get_embedding(p)
                if v:
                    self._cache[p] = v
                else:
                    logger.warning("RepE: Failed to get embedding for phrase: '%s'", p)

            if v:
                vectors.append(np.array(v, dtype=float))

        if not vectors:
            return np.zeros(1)

        # Stack and average to find the center of the concept
        return np.array(vectors, dtype=float).mean(axis=0)

    async def calibrate(self, force: bool = False):
        """Generates the Gestalt Axes (Good - Bad)."""
        current_hash = self._get_polarities_hash()
        cached_hash = self._cache.get("__polarities_hash__")

        if self.is_calibrated and current_hash == cached_hash and not force:
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
                logger.debug("   -> Calibrated Axis: %s", concept)
            else:
                logger.error(
                    "RepE: Skipped calibration for axis '%s' due to empty vectors.",
                    concept,
                )

        self._cache["__polarities_hash__"] = current_hash
        self._save_cache()
        self.is_calibrated = True

    def scan_thought(
        self, vector: List[float], text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Projects the thought onto all calibrated axes.
        Returns a 'Psychological Profile' including scores and rationales.
        """
        if not vector or not self.is_calibrated:
            # Handle lazy calibration if needed
            if not self.is_calibrated:
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self.calibrate())
                except RuntimeError:
                    asyncio.run(self.calibrate())
            return {"scores": {}, "rationale": None}

        thought_vec = np.array(vector, dtype=float)
        norm = np.linalg.norm(thought_vec)
        if norm > 0:
            thought_vec = thought_vec / norm

        scores: Dict[str, float] = {}
        for concept, axis in self.steering_axes.items():
            scores[concept] = round(float(np.dot(thought_vec, axis)), 3)

        # Generate rationale based on keyword matching if text is provided
        rationale = None
        if text:
            matches = []
            text_lower = text.lower()
            for concept, (neurotic_phrases, _) in self.polarities.items():
                if scores.get(concept, 0) < -0.1:  # Significant neurosis
                    for p in neurotic_phrases:
                        if p.lower() in text_lower:
                            matches.append(f"{concept}: '{p}'")
            if matches:
                rationale = "Psychological triggers detected: " + "; ".join(
                    list(set(matches))
                )

        trace_action(
            "REPE",
            "PSYCH_PROFILE",
            result=f"Neurosis Scan: {scores} | Rationale: {rationale}",
            tag="REPE",
        )

        return {"scores": scores, "rationale": rationale}


repe = GestaltMonitor()
