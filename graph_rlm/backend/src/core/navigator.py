"""
Navigator: The Engine of Curiousity for Graph-RLM.
Implements Intrinsic Motivation via Compression Progress and Causal Entropic Forces.
"""

import lzma
from typing import Any, Dict, List, Tuple

from .db import db
from .logger import get_logger
from .navigator_config import (
    COMPRESSION_WINDOW,
    EDGE_OF_CHAOS_LAMBDA_MAX,
    EDGE_OF_CHAOS_LAMBDA_MIN,
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
        except (RuntimeError, AttributeError, ValueError) as e:
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

        except (RuntimeError, AttributeError, ValueError) as e:
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

        # 0. Calculate Baseline Entropy (Current State)
        # We use a neutral anchor or the average of the last few steps
        baseline_s_tau = 0.5  # Default neutral anchor

        ranked = []
        for cand in candidates:
            # 1. Compression Progress (Learnability)
            r_t = self.compute_compression_progress(cand)

            # 2. Future Entropy (Freedom)
            current_s_tau = await self.estimate_future_entropy(cand)

            # 3. Langton's Lambda Filter (Edge of Chaos)
            # Class 4 behavior: where the system is most learnable and interesting.
            is_class_4 = (
                EDGE_OF_CHAOS_LAMBDA_MIN
                <= self._last_compression_ratio
                <= EDGE_OF_CHAOS_LAMBDA_MAX
            )

            # 4. Causal Entropic Force (F = T * grad(S_tau))
            # The force is the gradient of freedom: how much freedom this action ADDS.
            # grad_S = S_tau(candidate) - S_tau(baseline)
            grad_s = current_s_tau - baseline_s_tau

            # T (Temperature/Multiplier) is boosted at the Edge of Chaos
            temperature = 1.5 if is_class_4 else 1.0
            force = temperature * grad_s

            # Total Curiosity Score
            # Weighting: 40% Compression Progress, 60% Causal Entropic Force
            # We use a Sigmoid to squash scores into the 0.0 - 1.0 range
            # k=0.1 for a gentle slope centered at 0
            def sigmoid(x: float) -> float:
                import math

                return 1 / (1 + math.exp(-0.2 * x))

            raw_score = (r_t * 0.4) + (force * 0.6)
            score = sigmoid(raw_score)

            details = {
                "content": cand,
                "compression_progress": r_t,
                "future_entropy": current_s_tau,
                "causal_force": force,
                "is_class_4": is_class_4,
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
        # Query for successful thoughts
        # We look for thoughts with status='success' and substantial result content
        cypher = """
        MATCH (t:Thought {session_id: $sid, status: 'success'})
        WHERE t.result IS NOT NULL AND size(t.result) > 10
        RETURN t.id as id, t.prompt as prompt, t.result as result
        ORDER BY t.created_at DESC
        LIMIT 20
        """

        try:
            results = db.query(cypher, {"sid": session_id})
            patterns = []
            for row in results:
                # Handle both dict and object return types from DB wrapper
                r_prompt = (
                    row.get("prompt")
                    if isinstance(row, dict)
                    else getattr(row, "prompt", None)
                )
                r_result = (
                    row.get("result")
                    if isinstance(row, dict)
                    else getattr(row, "result", None)
                )
                r_id = (
                    row.get("id") if isinstance(row, dict) else getattr(row, "id", None)
                )

                if r_prompt and r_result:
                    patterns.append(
                        {
                            "id": r_id,
                            "prompt": r_prompt,
                            "result": r_result,
                            "compression_gain": 1.0,  # Placeholder until compute persist is active
                        }
                    )
            return patterns
        except (AttributeError, RuntimeError, KeyError, ValueError) as e:
            logger.warning(
                "Failed to extract learnable patterns for session %s: %s",
                session_id,
                e,
                exc_info=True,
            )
            return []


navigator = Navigator(sheaf_monitor=sheaf)
