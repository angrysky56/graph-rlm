"""
Navigator: The Engine of Curiousity for Graph-RLM.
Implements Intrinsic Motivation via Compression Progress and Causal Entropic Forces.
"""

import lzma
from typing import Any, Dict, List, Tuple

from .logger import get_logger
from .navigator_config import (
    COMPRESSION_WINDOW,
    LZMA_PRESET,
)

# Global instance
from .sheaf import sheaf

logger = get_logger("graph_rlm.navigator")


class Navigator:
    """
    The Navigator is the agent's active explorer.
    It ranks potential actions based on their 'Interestingness' (Curiosity Score).

    Metrics:
    1. Compression Progress: R(t) = C(D, t-1) - C(D, t)
    2. Causal Entropic Force: F = T * grad(S_tau)
    3. Topological Consistency: Via Sheaf Cohomology checks (delegated to Sheaf)
    """

    def __init__(self, sheaf_monitor=None):
        self.sheaf = sheaf_monitor
        self.history_buffer: List[str] = []
        self._last_compression_ratio = 1.0

    def compute_compression_size(self, data: str) -> int:
        """Calculates the compressed size of a string using LZMA."""
        if not data:
            return 0
        try:
            compressed = lzma.compress(data.encode("utf-8"), preset=LZMA_PRESET)
            return len(compressed)
        except Exception as e:
            logger.warning("Compression failed: %s", e)
            return len(data)

    def update_history(self, new_content: str):
        """Updates the history buffer and recalculates the baseline compression ratio."""
        self.history_buffer.append(new_content)
        if len(self.history_buffer) > COMPRESSION_WINDOW:
            self.history_buffer.pop(0)

        # Recalculate baseline
        full_text = "\n".join(self.history_buffer)
        raw_size = len(full_text)
        if raw_size > 0:
            comp_size = self.compute_compression_size(full_text)
            self._last_compression_ratio = comp_size / raw_size

    def compute_compression_progress(self, candidate_content: str) -> float:
        """
        Calculates the Intrinsic Reward R(t) = C(D, t-1) - C(D, t).
        Positive reward means the candidate makes the history MORE compressible (learnable).
        """
        if not self.history_buffer:
            return 0.0

        # Create hypothetical future history
        future_history = self.history_buffer + [candidate_content]
        full_text = "\n".join(future_history)
        raw_size = len(full_text)

        if raw_size == 0:
            return 0.0

        comp_size = self.compute_compression_size(full_text)
        new_ratio = comp_size / raw_size

        # Progress is the DROP in compression ratio (finding regularity)
        # R(t) = Ratio(t-1) - Ratio(t)
        progress = self._last_compression_ratio - new_ratio

        # amplify small distinct changes
        return progress * 100.0

    async def estimate_future_entropy(self, candidate_action: str) -> float:
        """
        Estimates S_tau (Future Freedom of Action) using Semantic Embeddings via RepE.

        We query the 'Freedom' axis (Exploration vs Restriction).
        Positive score = High Freedom (Exploration).
        Negative score = Low Freedom (Restriction).

        We normalize this to a 0.0 - 1.0 probability.
        """
        from .repe import repe

        if not candidate_action.strip():
            return 0.0

        try:
            # 1. Get embedding for candidate
            # We need the vector to scan it. RepE `scan_thought` takes a vector.
            # Ideally RepE would handle the embedding internally if passed text,
            # but current API takes vector.
            from .llm import llm

            candidate_vec = await llm.get_embedding(candidate_action)

            if not candidate_vec:
                return 0.5

            # 2. Scan with RepE
            profile = repe.scan_thought(candidate_vec)

            # 3. Extract Freedom Score
            # Range is roughly -1.0 to 1.0 (Cosine Similarity)
            freedom_score = profile.get("Freedom", 0.0)

            # Map -1..1 to 0..1
            # -1 (Restrictive) -> 0.0
            # +1 (Exploratory) -> 1.0
            normalized_entropy = (freedom_score + 1.0) / 2.0

            return max(0.0, min(1.0, normalized_entropy))

        except Exception as e:
            logger.warning("Semantic Entropy failed: %s", e)
            return 0.5

    async def compute_interest_gradient(
        self, candidates: List[str], history: List[Dict[str, Any]]
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Ranks candidate actions by their Curiosity Score.
        Returns list of (score, candidate_details).
        """
        if history and len(self.history_buffer) == 0:
            # Cold start: populate buffer from recent history
            for h in history[-COMPRESSION_WINDOW:]:
                self.history_buffer.append(h.get("result", "") or h.get("prompt", ""))

        ranked = []
        for cand in candidates:
            # 1. Compression Progress (Learnability)
            r_t = self.compute_compression_progress(cand)

            # 2. Future Entropy (Freedom)
            s_tau = await self.estimate_future_entropy(cand)

            # 3. Langton's Lambda Filter (Edge of Chaos)
            # We favor actions that are neither too ordered (r_t ~ 0) nor too random
            # This is implicitly handled by R(t) peaking at learnable complexity

            # Total Curiosity Score
            # Weight compression progress higher as it indicates "understanding"
            score = (r_t * 0.7) + (s_tau * 0.3)

            details = {
                "content": cand,
                "compression_progress": r_t,
                "future_entropy": s_tau,
                "score": score,
            }
            ranked.append((score, details))

        # Sort descending
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked

    def extract_learnable_patterns(
        self,
        session_id: str,  # pylint: disable=unused-argument
        min_compression_gain: float = 0.2,  # pylint: disable=unused-argument
    ) -> List[Dict[str, Any]]:
        """
        Retrospective analysis to find patterns that yielded high compression progress.
        Used by Dreamer to codify skills.
        """
        # Placeholder for full implementation which would query the graph for
        return []


navigator = Navigator(sheaf_monitor=sheaf)
