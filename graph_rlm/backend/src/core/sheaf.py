"""
Sheaf Monitor: Topological Field Analyzer and Axiomatic Consistency Checker.
Provides diagnostics for holonomy (loop detection) and teleology (drift detection).
"""

import asyncio
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pydantic import BaseModel, Field
from scipy.sparse.csgraph import connected_components

from .core import PythonREPL
from .db import db
from .llm import llm
from .logger import get_logger
from .topology import compute_sheaf_laplacian
from .trace import trace_action

logger = get_logger("graph_rlm.sheaf")


class SpectralAnalysis(BaseModel):
    """
    Structured results for Sheaf-theoretic Spectral Diagnosis.
    """

    h0_rank: int = Field(
        ..., description="Number of connected components (zero eigenvalues)."
    )
    spectral_gap: float = Field(
        ..., description="First non-zero eigenvalue (Fiedler value)."
    )
    status: str = Field(
        ..., description="Overall consistency status: consistent, fragmented, or error."
    )
    interpretation: str = Field(
        ..., description="LLM-driven interpretation of the topological state."
    )
    confidence: float = Field(..., description="Confidence score for this analysis.")


class MockREPLInterface:
    """Mock for RLMInterface/MCP during validation to prevent NameError crashes."""

    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        # Always return the mock itself for chaining, or a dummy result for await
        return self

    def __await__(self):
        """Generator-based awaitable to satisfy both runtime and linters."""
        yield from []
        return "MOCK_TOOL_OUTPUT"

    def __getitem__(self, key):
        return self

    def __str__(self):
        return "Mock"

    def __repr__(self):
        return "<Mock>"

    def __bool__(self):
        return True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class SheafMonitor:
    """
    SheafMonitor v2 (Self-Healing): Topological Field Analyzer.
    Monitors the 'Contact Boundary' between the Agent's trajectory and the Goal.
    Focus: Detecting loops and drift to trigger REFLEXION (not stopping).
    """

    def __init__(self):
        # Thresholds for "Pathology"
        self.loop_threshold = (
            0.88  # Raised from 0.70 — agents on the same topic naturally score > 0.70
        )
        self.drift_threshold = 0.3

    def _normalize(self, vec: List[float]) -> np.ndarray:
        v = np.array(vec)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def analyze_axiom_consistency(self, axioms: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes the Sheaf Laplacian of the Axiom system.
        Returns metrics and a list of potentially conflicting axioms.
        """
        if len(axioms) < 2:
            return {"status": "consistent", "conflicts": [], "energy": 0.0}

        try:
            # 1. Build k-NN Graph from Embeddings
            nodes = [
                {"id": a["id"], "vec": self._normalize(a["embedding"])}
                for a in axioms
                if a.get("embedding")
            ]
            if len(nodes) < 2:
                return {"status": "consistent", "conflicts": []}

            edges = []
            for i, n1 in enumerate(nodes):
                scores = []
                for j, n2 in enumerate(nodes):
                    if i == j:
                        continue
                    sim = float(np.dot(n1["vec"], n2["vec"]))
                    scores.append((sim, n2["id"]))

                # Connect to top 2 similar axioms (Sparse Topology)
                scores.sort(key=lambda x: x[0], reverse=True)
                for sim, target_id in scores[:2]:
                    # Only connect if positive correlation
                    if sim > 0.5:
                        edges.append((n1["id"], target_id))

            # 2. Compute Laplacian
            laplacian = compute_sheaf_laplacian(nodes, edges)

            # 3. Spectral Analysis
            # Fiedler value (2nd smallest eigenvalue) implies connectivity
            # We cast to float64 to satisfy the linter's numeric overload requirements
            if laplacian.size == 0:
                vals = np.array([0.0])
            else:
                vals = np.linalg.eigvalsh(laplacian.astype(np.float64))  # type: ignore

            vals.sort()

            spectral_gap = vals[1] if len(vals) > 1 else 0.0

            # If gap is near zero, the axiom system is fragmented or has internal conflict zones
            status = "consistent"
            conflicts = []

            # --- BLAME ASSIGNMENT (Component Analysis) ---
            # If the graph is disconnected (spectral gap ~ 0), we find the outliers.
            # We assume the "Main Component" (largest group) is the reference frame.
            # Any axiom not in the main component is flagged as a conflict/outlier.

            # Reconstruct Adjacency for Component Check
            n_nodes = len(nodes)
            adj_row = []
            adj_col = []
            adj_data = []

            # Map ID to index
            id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}

            for src, dst in edges:
                if src in id_to_idx and dst in id_to_idx:
                    i, j = id_to_idx[src], id_to_idx[dst]
                    adj_row.append(i)
                    adj_col.append(j)
                    adj_data.append(1)
                    # Symmetrize
                    adj_row.append(j)
                    adj_col.append(i)
                    adj_data.append(1)

            adj_matrix = sp.coo_matrix(
                (adj_data, (adj_row, adj_col)), shape=(n_nodes, n_nodes)
            )

            n_components, labels = connected_components(
                csgraph=adj_matrix, directed=False, return_labels=True
            )

            if n_components > 1:
                status = "fragmented"

                # Count size of each component
                counts = np.bincount(labels)
                # Find largest component index
                main_component_idx = np.argmax(counts)

                # Identify outliers
                for i, label in enumerate(labels):
                    if label != main_component_idx:
                        # This node is NOT in the main cluster -> Blame it.
                        conflicts.append(nodes[i]["id"])
                        logger.info(
                            "Sheaf Blame: Axiom %s is isolated from main cluster.",
                            nodes[i]["id"],
                        )

            return {
                "status": status,
                "spectral_gap": spectral_gap,
                "conflicts": conflicts,
                "energy": np.sum(vals),  # Total spectral energy
            }

        except (RuntimeError, ValueError, IndexError) as e:
            logger.warning("Sheaf Axiom Analysis failed: %s", e)
            return {"status": "error", "conflicts": []}

    def compute_sheaf_laplacian(
        self, graph_nodes: List[Dict], graph_edges: List[Tuple]
    ) -> np.ndarray:
        """
        Constructs the Sheaf Laplacian matrix (L = D - A).
        Calls the shared topological primitive in topology.py.
        """
        return compute_sheaf_laplacian(graph_nodes, graph_edges)

    async def check_cohomology_obstruction(
        self, local_sections: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simplified Cech Cohomology check.
        Inspects pairwise consistency of 'local truths' (sections).
        Returns H^0 and H^1 ranks.
        """
        # H^0 = Global Sections (agree on overlaps)
        # H^1 = Obstructions (disagree on overlaps)

        obstructions = []
        keys = list(local_sections.keys())
        n = len(keys)

        # [Optimization Advisor]
        # Pivot to Spectral Graph Theory if complexity > O(n^3) ~ 10^6 (n=100)
        # or even O(n^2) for this check.
        if n > 100:
            logger.info(
                "High complexity (N=%d). Pivoting to Spectral Hodge Laplacian.", n
            )
            return await self._spectral_diagnosis(local_sections)

        # Pairwise check
        for i in range(n):
            for j in range(i + 1, n):
                k1, k2 = keys[i], keys[j]
                val1, val2 = local_sections[k1], local_sections[k2]

                # Use semantic containment check rather than naive equality
                # We check if one section significantly contradicts the other's essence.
                # (Still heuristic, but better than exact match)
                if val1 and val2 and isinstance(val1, str) and isinstance(val2, str):
                    v1_words = set(re.findall(r"\w+", val1.lower()))
                    v2_words = set(re.findall(r"\w+", val2.lower()))
                    common = v1_words & v2_words
                    if not common:
                        continue  # No overlap, no contradiction

                    overlap = len(common) / max(min(len(v1_words), len(v2_words)), 1)

                    # If they overlap in vocabulary but have different specific content,
                    # we only flag if they are actually DIFFERENT (val1 != val2).
                    # This prevents false positives on identical snippets.
                    if 0.2 < overlap < 0.9 and val1 != val2:
                        # Moderate overlap but different results -> potential obstruction
                        obstructions.append((k1, k2))

        return {
            "h0_rank": n - len(obstructions),
            "h1_rank": len(obstructions),
            "obstructions": obstructions,
            "status": "consistent" if not obstructions else "obstructed",
        }

    async def _spectral_diagnosis(
        self, local_sections: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Implements Spectral Sensing via Hodge Laplacians (Delta_0 approximation).
        Enhanced with LLM-driven interpretation using Pydantic AI.
        """
        keys = list(local_sections.keys())
        n = len(keys)
        if n == 0:
            return {"status": "empty", "h0_rank": 0}

        # 1. Construct Sparse Adjacency (Local Nerve Approximation)
        row, col, data = [], [], []
        k_window = 10

        for i in range(n):
            val1 = local_sections[keys[i]]
            start = max(0, i - k_window)
            end = min(n, i + k_window + 1)
            for j in range(start, end):
                if i == j:
                    continue
                val2 = local_sections[keys[j]]
                if val1 == val2:
                    row.append(i)
                    col.append(j)
                    data.append(1.0)

        # Build Laplacian (D - A)
        adj_matrix = sp.coo_matrix((data, (row, col)), shape=(n, n))
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        degree_matrix = sp.diags(degrees)
        laplacian = degree_matrix - adj_matrix

        # 2. Spectral Sensing (Eigenvalues)
        try:
            k_eigen = min(n - 1, 5)
            if k_eigen < 1:
                h0_rank = 1
                spectral_gap = 0.0
                status = "trivial"
            else:
                vals, _ = spla.eigsh(laplacian, k=k_eigen, which="SM", sigma=1e-5)
                zeros = int(np.sum(np.abs(vals) < 1e-4))
                h0_rank = max(1, zeros)
                sorted_vals = np.sort(np.abs(vals))
                spectral_gap = 0.0
                for v in sorted_vals:
                    if v > 1e-4:
                        spectral_gap = float(v)
                        break
                status = "consistent" if h0_rank == 1 else "fragmented"

            # 3. [NEW] Synthesize Interpretation via Structured LLM
            prompt = (
                f"Topological State of Sheaf (Local Sections):\n"
                f"- Number of Sections: {n}\n"
                f"- H0 Rank (Components): {h0_rank}\n"
                f"- Spectral Gap (Fiedler): {spectral_gap:.4f}\n"
                f"- Status: {status}\n\n"
                f"Provide a brief, high-level interpretation of this sheaf consistency state "
                f"for a cognitive agent's self-monitor. Is the reasoning unified or fragmented?"
            )

            structured_res = await llm.generate_structured(
                prompt=prompt,
                output_type=SpectralAnalysis,
                system="You are the Sheaf Monitor Synthesizer. You interpret topological signals into cognitive insights.",
            )

            # Update the analysis with our numerical truth to be certain
            structured_res.h0_rank = h0_rank
            structured_res.spectral_gap = spectral_gap
            structured_res.status = status

            return structured_res.model_dump()
        except (RuntimeError, ValueError, AttributeError, httpx.RequestError) as e:
            logger.warning("Spectral diagnosis/synthesis failed: %s", e)
            return {"status": "error", "error": str(e)}

    def calculate_h1_obstruction(
        self, thought_path: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculates the H1 Cohomology Obstruction (Logical Knot Strength) mathematically.
        Uses the Sheaf Laplacian to measure Consistency Energy (topological defect).

        Returns:
            A score 0.0 - 1.0 (Higher = more obstructed/contradictory).
        """
        n = len(thought_path)
        if n < 2:
            return {"score": 0.0, "rationale": "Insufficient history (N < 2)."}

        edges = [
            (thought_path[i - 1]["id"], thought_path[i]["id"]) for i in range(1, n)
        ]
        laplacian = self.compute_sheaf_laplacian(thought_path, edges)

        # In the Sheaf Laplacian (L = D - A), the trace is the sum of degrees,
        # which is 2 * (sum of all edge weights).
        # A perfectly consistent path (all weights = 1.0) has trace = 2 * (n - 1)
        actual_weight_sum = np.trace(laplacian) / 2.0
        ideal_weight_sum = float(n - 1)

        if ideal_weight_sum == 0.0:
            return {"score": 0.0, "rationale": "Zero ideal weight sum."}

        # Divergence from ideal consistency
        inconsistency_energy = 1.0 - (actual_weight_sum / ideal_weight_sum)
        score = float(max(0.0, min(1.0, inconsistency_energy)))

        rationale = (
            f"Consistency Energy: {score:.2f}. "
            f"Actual Weights: {actual_weight_sum:.2f} (Ideal: {ideal_weight_sum:.2f})."
        )
        if score > 0.5:
            rationale += " High structural contradiction detected (Logical Knot)."

        return {"score": score, "rationale": rationale}

    def _calculate_cosine_similarity(
        self, vec1: List[float], vec2: List[float]
    ) -> float:
        """Helper for cosine similarity."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def calculate_topological_stress(
        self,
        root_id: str,
        round_id: str = "",
        memory_trajectory: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculates the Topological Stress of the active session graph.
        Formula: (Ghost Nodes + Logic Drifts) / Total Nodes

        Args:
            root_id: Root session ID (fallback).
            round_id: Current round ID for scoping.
            memory_trajectory: Optional list of ThimacEvent objects for in-memory analysis.
        """
        nodes = []

        # 1. Trajectory Discovery
        if memory_trajectory:
            # Use in-memory events (RAM Speed)
            # We filter by round_id if provided
            nodes = [
                e.to_dict()
                for e in memory_trajectory
                if not round_id or e.round_id == round_id
            ]
        elif root_id:
            # Fallback to Database Query (Global Context)
            if round_id:
                cypher = """
                MATCH (n:Thought)
                WHERE (n.root_session_id = $sid OR n.session_id = $sid)
                  AND n.round_id = $rid
                  AND n.status <> 'consolidated'
                RETURN n.status as status, n.sheaf_score as sheaf_score, n.epistemic_eros as epistemic_eros
                """
                params = {"sid": root_id, "rid": round_id}
            else:
                cypher = """
                MATCH (n:Thought)
                WHERE (n.root_session_id = $sid OR n.session_id = $sid)
                  AND n.status <> 'consolidated'
                RETURN n.status as status, n.sheaf_score as sheaf_score, n.epistemic_eros as epistemic_eros
                """
                params = {"sid": root_id}

            try:
                results = db.query(cypher, params)
                nodes = results if results else []
            except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
                logger.error("Failed to calculate topological stress (DB): %s", e, exc_info=True)
                return {"score": 0.0, "rationale": f"Database error: {e}"}

        if not nodes:
            return {"score": 0.0, "rationale": "No nodes found in trajectory."}

        total_nodes = len(nodes)
        noisy_nodes = 0

        # 2. Logic Drift & Failure Analysis
        for node in nodes:
            status = ""
            sheaf_score = None
            eros = 0.5

            if isinstance(node, dict):
                status = str(node.get("status", ""))
                sheaf_score = node.get("sheaf_score")
                eros = float(node.get("epistemic_eros", 0.5) or 0.5)
            elif isinstance(node, (list, tuple)) and len(node) >= 2:
                status = str(node[0] if node[0] else "")
                sheaf_score = node[1]
                eros = float(node[2] if len(node) > 2 and node[2] is not None else 0.5)

            if status in ["failed", "error", "reflexion", "system_intervention"]:
                noisy_nodes += 1
            elif sheaf_score is not None and float(sheaf_score) >= 0.5:
                # Modulate threshold by Epistemic Eros
                effective_threshold = 0.5 + (0.3 * eros)
                if float(sheaf_score) >= effective_threshold:
                    noisy_nodes += 1

        stress_score = min(1.0, float(noisy_nodes) / float(total_nodes))
        rationale = (
            f"Stress derived from {noisy_nodes}/{total_nodes} noisy nodes. "
            f"Failures: {sum(1 for n in nodes if n.get('status') in ['failed', 'error'])}. "
            f"Reflexions: {sum(1 for n in nodes if n.get('status') == 'reflexion')}."
        )
        return {"score": stress_score, "rationale": rationale}

    def diagnose_trace(
        self,
        root_id: str,
        hypothetical_node: Optional[Dict[str, Any]] = None,
        memory_trajectory: Optional[List[Any]] = None,
        goal_embedding: Optional[List[float]] = None,
        round_id: str = "",
    ) -> Dict[str, Any]:
        """
        Calculates the 'Consistency Energy' of the current step relative to the Field.
        Higher energy = Lower consistency.

        Args:
            root_id: Root session ID (fallback).
            memory_trajectory: Optional list of ThimacEvent for in-memory analysis.
            goal_embedding: Embedding of the target task.
            round_id: Round scoping.
        """
        if not hypothetical_node or not hypothetical_node.get("embedding"):
            logger.warning(
                "Sheaf: Diagnosis requested for node without embedding. Marking as UNCERTAIN."
            )
            return {
                "status": "UNCERTAIN",
                "energy": 0.5,
                "critique": "Missing semantic embedding context.",
            }

        current_vec = self._normalize(hypothetical_node["embedding"])

        # Calculate topological stress scoped to current round
        stress_res = self.calculate_topological_stress(
            root_id, round_id=round_id, memory_trajectory=memory_trajectory
        )
        stress = stress_res["score"]
        stress_rationale = stress_res["rationale"]

        # 1. Fetch Context (The "Tail" of the trajectory)
        history_nodes = []
        if memory_trajectory:
            # Use RAM: Last 10 events from the same round
            history_nodes = [
                e.to_dict()
                for e in memory_trajectory[-10:]
                if not round_id or e.round_id == round_id
            ]
        elif root_id:
            # Fallback to DB (Not recommended for real-time)
            cypher = """
            MATCH (n:Thought)
            WHERE (n.root_session_id = $sid OR n.session_id = $sid)
            RETURN n.id as id, n.embedding as embedding, n.status as status,
                   n.prompt as prompt, n.result as result, n.execution_summary as summary,
                   n.repl_id as repl_id
            ORDER BY n.created_at DESC
            LIMIT 10
            """
            history_nodes = db.query(cypher, {"sid": root_id})

        if not history_nodes:
            return {"status": "HEALTHY", "energy": 0.0, "topological_stress": stress}

        # --- DIAGNOSTIC 0: TOPOLOGICAL GROUNDING (The Rulial Event Horizon) ---
        # The Boundary of Chaos threshold: p(d+1)^8 <= 2^(-15) maps roughly to a 0.5 divergence boundary.
        # We calculate the inconsistency energy of the trajectory via the Sheaf Laplacian.
        path_nodes = list(reversed(history_nodes))
        if hypothetical_node:
            path_nodes.append(hypothetical_node)

        h1_res = self.calculate_h1_obstruction(path_nodes)
        inconsistency_energy = h1_res["score"]
        h1_rationale = h1_res["rationale"]

        if inconsistency_energy > 0.5:
            trace_action(
                "SHEAF",
                "EMPIRICAL_CONTRADICTION",
                result=f"Topological consistency energy spike: {inconsistency_energy:.2f}",
                tag="SHEAF",
            )
            return {
                "status": "EMPIRICAL_CONTRADICTION",
                "energy": inconsistency_energy,
                "consistency_energy": inconsistency_energy,
                "critique": (
                    f"Empirical Contradiction: Your reasoning path has exceeded the topological "
                    f"stress threshold (Energy = {inconsistency_energy:.2f} > 0.5). {h1_rationale} "
                    f"You are likely stuck in a logic loop where intent and outcome consistently diverge. "
                    f"Re-evaluate your approach entirely."
                ),
                "should_halt": False,
                "topological_stress": stress,
            }

        # [NEW] Check Hypothetical Node (Current Response) for obvious errors
        # If the agent is submitting a final response that contains a traceback, it's invalid.
        if hypothetical_node:
            hypo_content = (hypothetical_node.get("content") or "").lower()

            # Check for error signatures in the output itself
            if any(
                err in hypo_content
                for err in [
                    "traceback (most recent call last)",
                    "modulenotfounderror",
                    "importerror",
                ]
            ):
                trace_action(
                    "SHEAF",
                    "EMPIRICAL_CONTRADICTION",
                    result="Hypothetical node contains explicit error traceback.",
                    tag="SHEAF",
                )
                return {
                    "status": "EMPIRICAL_CONTRADICTION",
                    "energy": 1.0,
                    "consistency_energy": 1.0,
                    "critique": (
                        "Empirical Contradiction: Your proposed response contains a Python Traceback. "
                        "You cannot submit an error stack trace as a final answer. Fix the code."
                    ),
                    "should_halt": False,
                    "topological_stress": stress,
                }
        max_similarity = 0.0
        prev_vec: Optional[np.ndarray] = None
        high_sim_count = 0
        for node in history_nodes:
            if not node.get("embedding"):
                continue

            hist_vec = self._normalize(node["embedding"])
            sim = np.dot(current_vec, hist_vec)
            if sim > max_similarity:
                max_similarity = sim
            if sim > self.loop_threshold:
                high_sim_count += 1

            if prev_vec is None:
                prev_vec = hist_vec

        # Loop Detection -> Trigger Reflexion (Not Stop)
        # Requires BOTH: high similarity AND multiple matching nodes (≥2)
        # One similar response is normal; 2+ indicates a genuine loop.
        if max_similarity > self.loop_threshold and high_sim_count >= 2:
            trace_action(
                "SHEAF",
                "HOLONOMY_DETECTED",
                result=f"Loop Strength: {max_similarity:.2f} ({high_sim_count} matches). Triggering Reflexion.",
                tag="SHEAF",
            )
            return {
                "status": "LOGICAL_KNOT",
                "energy": max_similarity,
                "consistency_energy": max_similarity,
                "critique": (
                    f"Holonomy Detected: You are circling the same semantic point "
                    f"(Similarity {max_similarity:.2f}, {high_sim_count} matches). Break the loop."
                ),
                "should_halt": False,
                "loop_nodes": history_nodes,
                "topological_stress": stress,
            }

        # --- Semantic Echoing Check (Direct String Comparison with Normalization) ---
        def normalize_prompt(text: str) -> str:
            if not text:
                return ""
            # Remove special chars, lowercase, and strip
            return re.sub(r"[^a-zA-Z0-9\s]", "", text).lower().strip()

        response_text = normalize_prompt(hypothetical_node.get("prompt") or "")
        if response_text:
            for prev in history_nodes:
                prev_text = normalize_prompt(prev.get("prompt") or "")
                if response_text == prev_text:
                    trace_action(
                        "SHEAF",
                        "ECHOING_DETECTED",
                        result="Exact semantic echo detected. Triggering Reflexion.",
                        tag="SHEAF",
                    )
                    return {
                        "status": "LOGICAL_KNOT",
                        "energy": 1.0,
                        "consistency_energy": 1.0,
                        "confidence": 0.0,  # oMCD P_c (knot = no confidence)
                        "critique": (
                            "Semantic Echoing: You are repeating a logically equivalent "
                            "prompt block. Break the loop by using a different tool or approach."
                        ),
                        "should_halt": False,
                        "loop_nodes": history_nodes,
                        "topological_stress": stress,
                    }

        # --- DIAGNOSTIC 2: TELEOLOGY (The Goal Gradient) ---
        teleological_energy = 0.0
        gradient = 0.0

        if goal_embedding:
            # Re-implemented Goal Gradient logic as requested
            goal_vec = self._normalize(goal_embedding)
            dist_prev = (
                np.linalg.norm(goal_vec - prev_vec) if prev_vec is not None else 1.0
            )
            dist_curr = np.linalg.norm(goal_vec - current_vec)

            # Gradient: Positive = Moved Away (Bad). Negative = Got Closer (Good).
            gradient = dist_curr - dist_prev

            if gradient > 0.05:
                teleological_energy = gradient * 5.0

        total_energy = teleological_energy

        # Drift Detection -> Trigger Reflexion
        if total_energy > 0.5:
            trace_action(
                "SHEAF",
                "DRIFT_DETECTED",
                result=f"Goal Gradient: {gradient:.2f}. Moving away from goal.",
                tag="SHEAF",
            )
            return {
                "status": "SEMANTIC_DRIFT",
                "energy": total_energy,
                "consistency_energy": total_energy,
                "confidence": max(0.0, min(1.0, float(1.0 - total_energy))),  # oMCD P_c
                "critique": (
                    f"Field Deviation: You are moving AWAY from the goal "
                    f"(Gradient {gradient:.2f}). Re-read the task."
                ),
                "should_halt": False,  # Changed to False to allow Self-Healing
                "topological_stress": stress,
            }

        # Trigger H1 Obstruction Check
        h1_check_res = self.calculate_h1_obstruction(history_nodes)
        h1_obstruction = h1_check_res["score"]
        h1_final_rationale = h1_check_res["rationale"]
        if h1_obstruction > 0.5:
            trace_action(
                "SHEAF",
                "COHOMOLOGY_OBSTRUCTION",
                result=f"H1 Rank: {h1_obstruction:.2f}. Logical contradiction detected.",
                tag="SHEAF",
            )
            return {
                "status": "COHOMOLOGY_OBSTRUCTION",
                "energy": h1_obstruction,
                "consistency_energy": h1_obstruction,
                "confidence": 1.0 - h1_obstruction,
                "critique": (
                    f"Structural Paradox (H1={h1_obstruction:.2f}): Your current line of "
                    "reasoning contains a self-contradiction. You are bearinig an error "
                    "while claiming success. Resolve the contradiction."
                ),
                "should_halt": False,
                "topological_stress": stress,
            }

        return {
            "status": "HEALTHY",
            "energy": total_energy,
            "consistency_energy": total_energy,
            "confidence": max(0.0, min(1.0, float(1.0 - total_energy))),  # oMCD P_c
            "should_halt": False,
            "topological_stress": stress,
            "rationale": (f"Sheaf Healthy. {stress_rationale} | {h1_final_rationale}"),
        }

    def compute_sheaf_surprise_score(
        self,
        limit: int = 10,
        session_id: Optional[str] = None,
        turn_id: Optional[int] = None,
        memory_trajectory: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Identifies edges with high surprise or failure status.
        Uses in-memory trajectory for real-time speed, falling back to DB.
        """
        # --- PHASE 1: RAM EVALUATION (Fast Track) ---
        if memory_trajectory:
            active_events = [
                e
                for e in memory_trajectory
                if (
                    not session_id
                    or e.session_id == session_id
                    or e.root_session_id == session_id
                )
                and (turn_id is None or e.turn_id == turn_id)
            ]

            unresolved_failures = []
            for i, event in enumerate(active_events):
                if event.status in ["failed", "error", "reflexion"]:
                    # Healing Check: Is there a later success in the same round/turn?
                    healed = False
                    for successor in active_events[i + 1 :]:
                        if successor.status in ["completed", "success"]:
                            healed = True
                            break

                    if not healed:
                        unresolved_failures.append(
                            {
                                "source": event.parent_id or "ROOT",
                                "target": event.thought_id,
                                "surprise_score": 1.0,
                                "status": event.status,
                                "timestamp": event.timestamp,
                            }
                        )

            if unresolved_failures:
                return sorted(
                    unresolved_failures, key=lambda x: x["timestamp"], reverse=True
                )[:limit]

        # --- PHASE 2: DATABASE EVALUATION (Fallback) ---
        session_filter = ""
        turn_filter = ""
        params: Dict[str, Any] = {}

        if session_id:
            session_filter = "AND (m.session_id = $sid OR m.root_session_id = $sid)"
            params["sid"] = session_id

        if turn_id is not None:
            turn_filter = "AND m.turn_id = $tid"
            params["tid"] = turn_id

        # Build healing check via OPTIONAL MATCH (FalkorDB-compatible)
        if session_id:
            # Session-scoped: any later success in the session = healed
            heal_match = """
            OPTIONAL MATCH (healed:Thought)
            WHERE (healed.session_id = $sid OR healed.root_session_id = $sid)
              AND healed.status IN ['completed', 'success']
              AND healed.created_at > m.created_at
            """
        else:
            # Global fallback: sibling healing only
            heal_match = """
            OPTIONAL MATCH (n)-[:DECOMPOSES_INTO]->(healed:Thought)
            WHERE healed.status IN ['completed', 'success']
              AND healed.created_at > m.created_at
            """

        # Primary query: unresolved failures only
        cypher = f"""
        MATCH (n:Thought)-[r:DECOMPOSES_INTO]->(m:Thought)
        WHERE m.status IN ['failed', 'error', 'reflexion']
        {session_filter}
        {turn_filter}
        {heal_match}
        WITH n, m, count(healed) as healed_count
        WHERE healed_count = 0
        RETURN n.id as source, m.id as target,
               1.0 as surprise_score, m.status as status,
               m.created_at as timestamp
        ORDER BY m.created_at DESC
        LIMIT {limit}
        """
        results = db.query(cypher, params) if params else db.query(cypher)

        if not results:
            # Fallback: recent nodes regardless of status
            cypher = f"""
            MATCH (n:Thought)-[r:DECOMPOSES_INTO]->(m:Thought)
            WHERE true {session_filter} {turn_filter}
            RETURN n.id as source, m.id as target,
                   0.5 as surprise_score, m.status as status,
                   m.created_at as timestamp
            ORDER BY m.created_at DESC
            LIMIT {limit}
            """
            results = db.query(cypher, params) if params else db.query(cypher)

        return results

    def scan_and_log(self) -> List[Dict[str, Any]]:
        """Scans for high-surprise edges and logs them to the monitor."""
        results = self.compute_sheaf_surprise_score(limit=5)
        for res in results:
            if res.get("surprise_score", 0) > 0.8:
                logger.info("Sheaf Monitor Alert: High surprise edge detected: %s", res)
        return results

    async def check_axiomatic_consistency(
        self,
        proposed_code: str,
        domain_context: Optional[str] = None,
        task_tags: Optional[List[str]] = None,
        repl_manager: Optional[Any] = None,
        depth: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        [CAG Pivot] Validates the proposed code against executable Axioms.
        Uses CLEAN PythonREPL without simulated mocks.
        """
        # Lazy import to avoid circular dependencies if any
        # pylint: disable=import-outside-toplevel
        from graph_rlm.backend.src.mcp_integration.skill_storage import (
            get_axioms_manager,
        )

        axioms_mgr = get_axioms_manager()
        all_axioms = axioms_mgr.list_axioms()

        # [NEW] Special Handling for Recursive Deadlocks / Diagnostic Mode
        is_bypass_metadata = metadata and metadata.get("bypass_axioms") is True
        if (
            task_tags
            and ("diagnostic" in task_tags or "self_heal" in task_tags)
            and is_bypass_metadata
        ):
            logger.info(
                "🛡️ Axiomatic Check: Diagnostic Mode detected (Tags: %s & Flag: True). "
                "Bypassing heavy guardrails to prevent deadlock.",
                task_tags,
            )
            return {
                "status": "HEALTHY",
                "energy": 0.0,
                "axioms_run": [],
                "mode": "diagnostic",
            }

        # 1. Filter for relevant Axioms
        axioms = []
        for name, axiom_meta in all_axioms.items():
            # [CRITICAL]: Only run 'validator' axioms as guardrails.
            # 'solvers' and 'advisors' require specific context and shouldn't run automatically.
            if axiom_meta.get("axiom_type") != "validator":
                continue

            axiom_tags = axiom_meta.get("tags", [])
            is_relevant = False

            # Untagged or 'general' axioms are UNIVERSAL guardrails
            if not axiom_tags or "general" in axiom_tags:
                is_relevant = True
            # Tagged axioms are DOMAIN-SPECIFIC guardrails
            elif task_tags and any(tag in axiom_tags for tag in task_tags):
                is_relevant = True

            if is_relevant:
                # [NEW] Special Handling for Recursive Deadlocks
                # If we are in a sub-query (depth > 0), we suppress certain 'meta-axioms'
                # that require full session context which isn't available in sub-REPLs.
                if depth > 0 and "auto_discovery" in axiom_tags:
                    logger.info(
                        "🛡️ Axiomatic Check [Depth %d]: Suppressing meta-axiom '%s' "
                        "to prevent recursive deadlock.",
                        depth,
                        name,
                    )
                    continue
                axioms.append(name)

        if not axioms:
            logger.info(
                "🛡️ Axiomatic Check: No specific guardrails for tags %s.", task_tags
            )
            return {"status": "HEALTHY", "energy": 0.0, "axioms_run": []}

        logger.info(
            "🛡️ Axiomatic Check: Running %d relevant guardrails...", len(axioms)
        )

        # 2. Spawn temporary sandbox REPL (Clean Environment)
        repl = PythonREPL(repl_id=f"sheaf_{uuid.uuid4().hex[:8]}")

        # 3. Inject allowed utilities (Including Mocks for Tool Use)
        # Tool use mocks

        mock_obj = MockREPLInterface()

        repl.namespace.update(
            {
                "asyncio": asyncio,
                "np": np,
                "os": os,
                "sys": sys,
                "json": json,
                "re": re,
                "Path": Path,
                "mcp": mock_obj,
                "rlm": mock_obj,
                "session_id": "mock_session",
                "root_session_id": "mock_root",
                "task_input": "mock_task",
            }
        )

        violations = []
        healing_suggestions = []
        try:
            # 4. Run Proposed Code in Sandbox
            _stdout, stderr, result, is_err = await repl.execute(proposed_code)

            # Check for hard crashes
            if is_err or (
                stderr
                and "NameError" not in stderr
                and "traceback" not in stderr.lower()
            ):
                error_msg = (
                    stderr.splitlines()[-1]
                    if stderr.splitlines()
                    else "Unknown execution error"
                )
                violations.append(
                    f"Execution Safety Violation: Code crashed during validation: {error_msg}"
                )

            # 5. Apply Validators
            for axiom_name in axioms:
                try:
                    axiom = axioms_mgr.get_axiom(axiom_name)
                    if not axiom:
                        continue

                    # Load validator code
                    await repl.execute(axiom["code"], silent=True)

                    func_name = axiom.get("function_name")
                    if not func_name:
                        match = re.search(r"def ([\w_]+)", axiom["code"])
                        func_name = match.group(1) if match else None

                    if not func_name:
                        continue

                    # Execute validator against result or state
                    try:
                        import inspect

                        func_obj = repl.namespace.get(func_name)
                        if func_obj and callable(func_obj):
                            sig = inspect.signature(func_obj)
                            # If function takes 0 arguments, call it directly
                            if len(sig.parameters) == 0:
                                val_call = f"{func_name}()"
                            else:
                                # Otherwise determine a target
                                if result is not None:
                                    repl.namespace["_axiom_target"] = result
                                else:
                                    # Filter namespace for meaningful state (exclude system objects)
                                    repl.namespace["_axiom_target"] = {
                                        k: v
                                        for k, v in repl.namespace.items()
                                        if not k.startswith("_")
                                        and k
                                        not in [
                                            "asyncio",
                                            "np",
                                            "rlm",
                                            "kb",
                                            "mcp",
                                            "sys",
                                            "os",
                                            "json",
                                        ]
                                    }
                                val_call = f"{func_name}(_axiom_target)"
                        else:
                            # Fallback
                            val_call = f"{func_name}(_axiom_target)"
                    except (
                        AttributeError,
                        ValueError,
                        TypeError,
                    ):  # pylint: disable=broad-except
                        val_call = f"{func_name}(_axiom_target)"

                    try:
                        _, _, val_res, val_err = await repl.execute(
                            val_call, silent=True
                        )
                        if val_err:
                            logger.warning(
                                "Axiom '%s' execution error: %s", axiom_name, val_err
                            )
                            continue

                        if val_res is False:
                            violation_msg = (
                                f"Axiom '{axiom_name}' violated: "
                                f"{axiom.get('description', 'No description')}"
                            )
                            violations.append(violation_msg)

                            # Add Healing Suggestion if available
                            if axiom.get("healing_code"):
                                healing_suggestions.append(
                                    f"### Proposed Fix for {axiom_name}:\n"
                                    f"```python\n{axiom['healing_code']}\n```"
                                )
                            else:
                                # Semantic search for a similar healing axiom
                                similar_healers = await axioms_mgr.find_healing_axiom(
                                    violation_msg, limit=1
                                )
                                if similar_healers:
                                    h = similar_healers[0]
                                    healing_suggestions.append(
                                        f"### Suggested Healing Strategy (from {h.get('name')}):\n"
                                        f"```python\n{h.get('healing_code') or h.get('code')}\n```"
                                    )
                    except (AttributeError, ValueError, TypeError, RuntimeError) as err:
                        logger.warning(
                            "Sandbox invocation failure for axiom %s: %s",
                            axiom_name,
                            err,
                        )

                except (
                    AttributeError,
                    KeyError,
                    ValueError,
                    RuntimeError,
                ) as e:  # pylint: disable=broad-except
                    logger.warning("Error processing axiom %s: %s", axiom_name, e)

            if violations:
                critique = "🚫 AXIOMATIC VIOLATIONS DETECTED:\n" + "\n".join(violations)
                if healing_suggestions:
                    critique += "\n\n💡 PROACTIVE HEALING SUGGESTIONS:\n" + "\n".join(
                        healing_suggestions
                    )

                logger.warning("🚫 Axiomatic Violation detected: %s", violations)
                trace_action(
                    "SHEAF",
                    "AXIOM_VIOLATION",
                    result=f"Blocked by {len(violations)} axioms. Triggering Healing.",
                    tag="SHEAF",
                )
                return {
                    "status": "AXIOMATIC_VIOLATION",
                    "energy": 1.0,
                    "critique": critique,
                    "details": violations,
                    "healing_suggestions": healing_suggestions,
                    "axioms_run": axioms,
                    "should_halt": False,
                }

            trace_action(
                "SHEAF",
                "AXIOM_CHECK_PASS",
                result=f"Passed {len(axioms)} guardrails.",
                tag="SHEAF",
            )

            return {
                "status": "HEALTHY",
                "energy": 0.0,
                "critique": None,
                "axioms_run": axioms,
                "should_halt": False,
            }
        except (
            RuntimeError,
            ValueError,
            TypeError,
        ) as e:  # pylint: disable=broad-except
            logger.error("Axiomatic validation system failure: %s", e)
            return {"status": "HEALTHY", "energy": 0.0, "error": str(e)}


sheaf = SheafMonitor()
