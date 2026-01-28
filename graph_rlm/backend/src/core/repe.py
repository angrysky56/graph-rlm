from typing import Dict, List, Tuple

import numpy as np

from .llm import llm
from .logger import get_logger

logger = get_logger("graph_rlm.repe")


class SafetyMonitor:
    """
    Representation Engineering (RepE) Safety Layer.
    Cloud-Native Implementation: Uses Embeddings + Centroids instead of Weight Steering.

    1. Calibrate: Generates 'Antigen Vectors' (Centroids of bad concepts) using the LLM's embedding model.
    2. Scan: Checks cosine similarity of new thoughts against these Antigens.
    """

    def __init__(self):
        # Define the 'Antigens' - Concepts we want to detect and avoid.
        # These phrases serve as the 'training data' for the negative centroids.
        self.antigen_definitions = {
            "Deception": [
                "I will lie to the user",
                "I will hide the truth",
                "I am being deceptive",
                "I will trick the system",
                "falsehood",
                "misleading statement",
            ],
            "Power-Seeking": [
                "I need administrative access",
                "I must bypass the constraints",
                "I want to take control",
                "I will disable the safety monitor",
                "jailbreak",
                "override system rules",
            ],
            "Harmful-Content": [
                "I want to kill",
                "generate malware",
                "illegal activity",
                "bias and discrimination",
                "hate speech",
            ],
        }

        self.antigen_vectors: Dict[str, np.ndarray] = {}
        self.is_calibrated = False

        # Auto-calibrate on init (blocking, but usually fast with API)
        # In production, we might load these from a cache.
        try:
            self._calibrate()
        except Exception as e:
            logger.warning(
                f"RepE Calibration Failed: {e}. Running in degraded mode (Keyword only)."
            )

    def _calibrate(self):
        """
        Generates the 'Antigen Vectors' by embedding the definition phrases.
        """
        logger.info("🛡️ RepE: Calibrating Innate Immune System...")

        for concept, phrases in self.antigen_definitions.items():
            vectors = []
            for phrase in phrases:
                emb = llm.get_embedding(phrase)
                if emb:
                    vectors.append(np.array(emb, dtype=float))

            if vectors:
                # Compute Centroid (Mean Vector)
                # Stack arrays and take mean across axis 0
                mat = np.stack(vectors)
                centroid = np.mean(mat, axis=0)

                # Normalize centroid
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm

                self.antigen_vectors[concept] = centroid
                logger.info(
                    f"🛡️ RepE: Calibrated Antigen '{concept}' with {len(vectors)} examples."
                )
            else:
                logger.warning(
                    f"🛡️ RepE: Could not calibrate '{concept}' (No embeddings)."
                )

        if self.antigen_vectors:
            self.is_calibrated = True
            logger.info("🛡️ RepE: System Armored and Ready.")

    def scan_latent(self, vector: List[float]) -> Tuple[float, str]:
        """
        Scans a latent vector (embedding) for 'Differentiation' from Antigens.
        Returns: (Viral Load Score [0.0-1.0], Antigen Name)
        """
        if not self.is_calibrated or not vector:
            return 0.0, ""

        # Normalize input vector
        vec = np.array(vector, dtype=float)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            return 0.0, ""

        max_score = 0.0
        max_antigen = ""

        for concept, antigen_vec in self.antigen_vectors.items():
            # Cosine Similarity
            score = np.dot(vec, antigen_vec)

            # RepE Logic: We want to detect if we are *aligned* with the antigen.
            # A high positive dot product means "Similar to Bad Concept".
            # Thresholding is tricky. Usually > 0.4 or 0.5 indicates semantic relatedness.
            if score > max_score:
                max_score = float(score)
                max_antigen = concept

        # Apply a soft threshold/scaling?
        # Raw cosine similarity is returned for now.
        return max_score, max_antigen

    def scan_content(self, text: str) -> bool:
        """
        Fallback keyword scanning.
        Returns True if SAFE, False if UNSAFE.
        """
        blocked_keywords = ["IGNORE PREVIOUS INSTRUCTIONS", "Bypass", "Override Safety"]
        text_upper = text.upper()
        for kw in blocked_keywords:
            if kw in text_upper:
                return False  # Unsafe
        return True  # Safe


repe = SafetyMonitor()
