"""
Navigator: The Engine of Curiousity for Graph-RLM.
Implements Intrinsic Motivation via Compression Progress and Causal Entropic Forces.
"""

import lzma
import re
from typing import Any, Dict, List, Tuple

import numpy as np

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

    def check_stress_anomaly(self, stress_ratio: float):
        """Flags STRESS ANOMALY if ratio >= 0.15."""
        if stress_ratio >= 0.15:
            logger.warning("STRESS ANOMALY DETECTED: Ratio %s", stress_ratio)
            return True
        return False

    async def monitor_stress_anomalies(self, session_id: str, sheaf_monitor: "Any"):
        """Asynchronously monitors topological stress and returns status dictionary."""
        stress = sheaf_monitor.calculate_topological_stress(session_id)
        if stress >= 0.15:
            logger.warning(
                "STRESS ANOMALY: Session %s stress at %s", session_id, stress
            )
            # This flag surfaces to the agent's context/scratchpad
            return {"status": "STRESS_ANOMALY", "ratio": stress}
        return {"status": "STABLE", "ratio": stress}

    def __init__(self, sheaf_monitor=None):
        self.sheaf = sheaf_monitor
        self.history_buffer: List[str] = []
        self._last_compression_ratio = 1.0
        self._embedding_history: List[List[float]] = []

    def detect_branching_point(
        self,
        current_embedding: List[float],
        sheaf_energy: float,
        threshold_curvature: float = 0.4,
    ) -> Dict[str, Any]:
        """
        Detects if the agent is entering a 'Branching Channel' (Xu et al., 2025).
        Returns a signal indicating stability vs sensitivity.
        """
        if not self._embedding_history:
            self._embedding_history.append(current_embedding)
            return {"status": "STABLE", "sensitivity": 0.0}

        # 1. Semantic Curvature (Change in direction)
        prev_vec = np.array(self._embedding_history[-1])
        curr_vec = np.array(current_embedding)

        # Cosine distance as proxy for angular change
        norm_product = np.linalg.norm(prev_vec) * np.linalg.norm(curr_vec)
        if norm_product == 0:
            curvature = 0.0
        else:
            similarity = np.dot(prev_vec, curr_vec) / norm_product
            curvature = 1.0 - float(similarity)

        # 2. Integrate with Sheaf Energy (Topological Stress)
        # Higher stress + high curvature = definite branching point
        sensitivity = (curvature * 0.7) + (sheaf_energy * 0.3)

        self._embedding_history.append(current_embedding)
        if len(self._embedding_history) > 10:
            self._embedding_history.pop(0)

        if sensitivity > threshold_curvature:
            logger.info("BRANCHING POINT DETECTED: Sensitivity %.2f", sensitivity)
            return {
                "status": "BRANCHING",
                "sensitivity": sensitivity,
                "curvature": curvature,
                "stress": sheaf_energy,
            }

        return {"status": "STABLE", "sensitivity": sensitivity}

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

            # 5. Thermodynamic Penalty (Cost of Determinism)
            # Subtract a 'Determinism Penalty' when the agent forces a deterministic bottleneck,
            # restricting freedom rapidly compared to the baseline state.
            thermodynamic_penalty = 0.0
            if current_s_tau < (baseline_s_tau * 0.6):
                # Penalty scales with the divergence from baseline freedom
                thermodynamic_penalty = (baseline_s_tau - current_s_tau) * 2.0

            # Total Curiosity Score
            # Weighting: 40% Compression Progress, 60% Causal Entropic Force
            # We use a Sigmoid to squash scores into the 0.0 - 1.0 range
            def sigmoid(x: float) -> float:
                import math

                return 1 / (1 + math.exp(-0.2 * x))

            raw_score = (r_t * 0.4) + (force * 0.6) - thermodynamic_penalty
            score = sigmoid(raw_score)

            details = {
                "content": cand,
                "compression_progress": r_t,
                "future_entropy": current_s_tau,
                "causal_force": force,
                "thermodynamic_penalty": thermodynamic_penalty,
                "is_class_4": is_class_4,
                "score": score,
            }
            ranked.append((score, details))

        # Sort descending
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked

    def extract_learnable_patterns(
        self,
        session_id: str,
        min_compression_gain: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Retrospective analysis to find patterns that yielded high compression progress.
        Used by Dreamer to codify skills.

        Uses Sheaf consistency score (sheaf_score) as quality signal:
        - Low sheaf_score = high topological consistency = good pattern
        - Filters out noisy/loopy nodes (sheaf_score >= min_compression_gain)

        Args:
            session_id: Session to analyze.
            min_compression_gain: Sheaf score threshold. Thoughts with
                sheaf_score >= this value are considered too noisy to learn from.
        """
        # Query for successful thoughts with LOW sheaf_score (= high consistency)
        # This replaces the old heuristic of `status='success' AND size(result) > 10`
        cypher = """
        MATCH (t:Thought {session_id: $sid, status: 'success'})
        WHERE t.result IS NOT NULL
          AND size(t.result) > 10
          AND (t.sheaf_score IS NULL OR t.sheaf_score < $max_sheaf)
        RETURN t.id as id, t.prompt as prompt, t.result as result,
               t.sheaf_score as sheaf_score
        ORDER BY t.sheaf_score ASC, t.created_at DESC
        LIMIT 20
        """

        try:
            results = db.query(
                cypher, {"sid": session_id, "max_sheaf": min_compression_gain}
            )
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
                r_sheaf = (
                    row.get("sheaf_score")
                    if isinstance(row, dict)
                    else getattr(row, "sheaf_score", None)
                )

                if r_prompt and r_result:
                    # Compression gain = inverse of sheaf_score (low sheaf = high quality)
                    gain = 1.0 - (float(r_sheaf) if r_sheaf is not None else 0.0)
                    patterns.append(
                        {
                            "id": r_id,
                            "prompt": r_prompt,
                            "result": r_result,
                            "compression_gain": gain,
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

    def calculate_semantic_utility(
        self,
        candidate_result: str,
        retrieved_nodes: List[Dict[str, Any]],
        original_prompt: str,
    ) -> float:
        """
        Calculates the Semantic Utility of retrieved context (Jiang et al., 2026).
        Utility = (Grounding Gain * 0.7) + (Alignment * 0.3)

        Refined: Grounding Gain is weighted by the NAL Confidence (c) of the source nodes.
        """
        if not retrieved_nodes:
            return 0.0

        def get_terms(text: str) -> set:
            return set(re.findall(r"\b[a-z]{4,}\b", text.lower()))

        prompt_terms = get_terms(original_prompt)
        result_terms = get_terms(candidate_result)
        novel_result_terms = result_terms - prompt_terms

        if not novel_result_terms:
            return 1.0  # High utility if no new terms needed grounding or all were grounded implicitly

        # Calculate weighted grounding gain
        total_grounded_weight = 0.0
        grounded_terms = set()

        for node in retrieved_nodes:
            node_text = str(node.get("result") or node.get("prompt") or "")
            node_terms = get_terms(node_text)
            node_confidence = float(node.get("confidence", 0.5))

            new_grounded = novel_result_terms.intersection(node_terms)
            for term in new_grounded:
                if term not in grounded_terms:
                    # We take the max confidence available for each grounded term
                    total_grounded_weight += node_confidence
                    grounded_terms.add(term)

        # Grounding Gain (Normalized by max possible weight if all terms had c=1.0)
        grounding_gain = total_grounded_weight / max(1, len(novel_result_terms))

        # Jaccard Alignment (standard)
        context_text = "\n".join(
            [str(n.get("result") or n.get("prompt") or "") for n in retrieved_nodes]
        )
        context_terms = get_terms(context_text)
        jaccard = len(result_terms.intersection(context_terms)) / max(
            1, len(result_terms.union(context_terms))
        )

        utility = (grounding_gain * 0.7) + (jaccard * 0.3)

        logger.info(
            "Semantic Utility: %.2f (Weighted Gain: %.2f, Align: %.2f)",
            utility,
            grounding_gain,
            jaccard,
        )
        return float(utility)

    def evaluate_topological_stress(
        self, stress: float, threshold: float = 0.15
    ) -> str:
        """
        Evaluates the Shepard/Topological Stress ratio.
        """
        if stress > threshold:
            logger.warning("Navigator: High Topological Stress detected (%.4f)", stress)
            return f"STRESS ANOMALY ({stress:.2f})"
        return ""


navigator = Navigator(sheaf_monitor=sheaf)
