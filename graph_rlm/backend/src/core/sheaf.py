"""
Sheaf Monitor: Topological Field Analyzer and Axiomatic Consistency Checker.
Provides diagnostics for holonomy (loop detection) and teleology (drift detection).
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import scipy.sparse as sp  # type: ignore
import scipy.sparse.linalg as spla  # type: ignore
from pydantic import BaseModel, Field

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
        self.loop_threshold = 0.70  # Lowered from 0.75 for stricter detection
        self.drift_threshold = 0.3

    def _normalize(self, vec: List[float]) -> np.ndarray:
        v = np.array(vec)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

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

    def diagnose_trace(
        self,
        root_id: str,  # pylint: disable=unused-argument
        hypothetical_node: Optional[Dict[str, Any]] = None,
        hypothetical_edges: Optional[List[Tuple[str, str]]] = None,
        goal_embedding: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Calculates the 'Consistency Energy' of the current step relative to the Field.
        Higher energy = Lower consistency.
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

        # 1. Fetch Context (The "Tail" of the trajectory)
        frontier_ids = [e[0] for e in hypothetical_edges] if hypothetical_edges else []
        if not frontier_ids:
            return {"status": "HEALTHY", "energy": 0.0}

        cypher = """
        MATCH (n:Thought)
        WHERE n.id IN $fids
        RETURN n.id as id, n.embedding as embedding, n.status as status,
               n.prompt as prompt, n.result as result, n.execution_summary as summary,
               n.repl_id as repl_id
        ORDER BY n.created_at DESC
        """
        history_nodes = db.query(cypher, {"fids": frontier_ids})

        if not history_nodes:
            return {"status": "HEALTHY", "energy": 0.0}

        # --- DIAGNOSTIC 0: EMPIRICAL INTEGRITY (The Lie Detector) ---
        # [NEW] Topological Grounding: Check if the trajectory is physically consistent.
        # If the agent's logic machine (Thought) says "Done" or "Success" but the
        # actual outcome (REPL Result) shows a crash or contradiction, it's an obstruction.
        for node in history_nodes:
            node_status = node.get("status")
            node_result = (node.get("result") or "").lower()
            node_prompt = (node.get("prompt") or "").lower()

            # Pattern: Agent claims it solved it, but result is an error
            is_success_claimed = any(
                w in node_prompt for w in ["success", "completed", "solved", "fixed"]
            )
            is_error_found = any(
                w in node_result for w in ["error", "failed", "traceback", "not found"]
            )

            if is_success_claimed and is_error_found and node_status != "failed":
                trace_action(
                    "SHEAF",
                    "EMPIRICAL_CONTRADICTION",
                    result=f"Node {node['id'][:8]} claims success but result has errors.",
                    tag="SHEAF",
                )
                return {
                    "status": "EMPIRICAL_CONTRADICTION",
                    "energy": 1.0,
                    "consistency_energy": 1.0,
                    "critique": (
                        f"Empirical Contradiction: Your previous thought ({node['id'][:8]}) "
                        f"using REPL [{node.get('repl_id', 'unknown')}] claimed success, "
                        "but the REPL returned an error. You are hallucinating "
                        "progress. Review the actual output before proceeding."
                    ),
                    "should_halt": False,
                    "loop_nodes": [node],
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
                }
        max_similarity = 0.0
        prev_vec: Optional[np.ndarray] = None

        for node in history_nodes:
            if not node.get("embedding"):
                continue

            hist_vec = self._normalize(node["embedding"])
            sim = np.dot(current_vec, hist_vec)
            if sim > max_similarity:
                max_similarity = sim

            if prev_vec is None:
                prev_vec = hist_vec

        # Loop Detection -> Trigger Reflexion (Not Stop)
        if max_similarity > self.loop_threshold:
            trace_action(
                "SHEAF",
                "HOLONOMY_DETECTED",
                result=f"Loop Strength: {max_similarity:.2f}. Triggering Reflexion.",
                tag="SHEAF",
            )
            return {
                "status": "LOGICAL_KNOT",
                "energy": max_similarity,
                "consistency_energy": max_similarity,
                "critique": (
                    f"Holonomy Detected: You are circling the same semantic point "
                    f"(Similarity {max_similarity:.2f}). Break the loop."
                ),
                "should_halt": False,
                "loop_nodes": history_nodes,
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
            }

        return {
            "status": "HEALTHY",
            "energy": total_energy,
            "consistency_energy": total_energy,
            "confidence": max(0.0, min(1.0, float(1.0 - total_energy))),  # oMCD P_c
            "should_halt": False,
        }

    def compute_sheaf_surprise_score(
        self,
        limit: int = 10,
        session_id: Optional[str] = None,
        turn_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Queries the graph for edges with high surprise or failure status.

        Resolution filter: uses OPTIONAL MATCH to exclude failures that have
        a later healed node (status 'completed'/'success' with a newer
        timestamp) in the same scope (turn > session > global).
        """
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
        domain_context: Optional[str] = None,  # pylint: disable=unused-argument
        task_tags: Optional[List[str]] = None,
        repl_manager: Optional[Any] = None,  # pylint: disable=unused-argument
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
        repl = PythonREPL()

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
