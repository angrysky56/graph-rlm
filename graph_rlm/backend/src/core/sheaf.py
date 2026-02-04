import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .core import PythonREPL
from .db import db
from .logger import get_logger

logger = get_logger("graph_rlm.sheaf")


class SheafMonitor:
    """
    SheafMonitor v2 (Self-Healing): Topological Field Analyzer.
    Monitors the 'Contact Boundary' between the Agent's trajectory and the Goal.
    Focus: Detecting loops and drift to trigger REFLEXION (not stopping).
    """

    def __init__(self):
        # Thresholds for "Pathology"
        self.loop_threshold = 0.85
        self.drift_threshold = 0.3

    def _normalize(self, vec: List[float]) -> np.ndarray:
        v = np.array(vec)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def diagnose_trace(
        self,
        root_id: str,
        hypothetical_node: Optional[Dict[str, Any]] = None,
        hypothetical_edges: Optional[List[Tuple[str, str]]] = None,
        goal_embedding: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Calculates the 'Consistency Energy' of the current step relative to the Field.
        Higher energy = Lower consistency.
        """
        if not hypothetical_node or not hypothetical_node.get("embedding"):
            return {"status": "HEALTHY", "energy": 0.0}

        current_vec = self._normalize(hypothetical_node["embedding"])

        # 1. Fetch Context (The "Tail" of the trajectory)
        frontier_ids = [e[0] for e in hypothetical_edges] if hypothetical_edges else []
        if not frontier_ids:
            return {"status": "HEALTHY", "energy": 0.0}

        cypher = """
        MATCH (n:Thought)
        WHERE n.id IN $fids
        RETURN n.id as id, n.embedding as embedding, n.status as status
        ORDER BY n.created_at DESC LIMIT 5
        """
        history_nodes = db.query(cypher, {"fids": frontier_ids})

        if not history_nodes:
            return {"status": "HEALTHY", "energy": 0.0}

        # --- DIAGNOSTIC 1: HOLONOMY (The Loop Detector) ---
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
            return {
                "status": "LOGICAL_KNOT",
                "energy": max_similarity,
                "critique": f"Holonomy Detected: You are circling the same semantic point (Similarity {max_similarity:.2f}). Break the loop.",
                "should_halt": False,
                "loop_nodes": history_nodes,  # [Dreamer Link] Pass context for deep analysis
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
            return {
                "status": "SEMANTIC_DRIFT",
                "energy": total_energy,
                "critique": f"Field Deviation: You are moving AWAY from the goal (Gradient {gradient:.2f}). Re-read the task.",
                "should_halt": False,  # Changed to False to allow Self-Healing
            }

        return {
            "status": "HEALTHY",
            "energy": total_energy,
            "consistency_energy": total_energy,
            "should_halt": False,
        }

    def compute_sheaf_surprise_score(
        self, limit: int = 10, session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries the graph for edges with high surprise or failure status.
        """
        session_filter = ""
        params = {}
        if session_id:
            session_filter = "AND (m.session_id = $sid OR m.root_session_id = $sid)"
            params["sid"] = session_id

        cypher = f"""
        MATCH (n:Thought)-[r:DECOMPOSES_INTO]->(m:Thought)
        WHERE (m.status = 'failed' OR m.status = 'error' OR m.status = 'reflexion') {session_filter}
        RETURN n.id as source, m.id as target, 1.0 as surprise_score, m.status as status
        LIMIT {limit}
        """
        results = db.query(cypher, params) if params else db.query(cypher)

        if not results:
            cypher = f"""
            MATCH (n:Thought)-[r:DECOMPOSES_INTO]->(m:Thought)
            WHERE true {session_filter}
            RETURN n.id as source, m.id as target, 0.5 as surprise_score, m.status as status
            ORDER BY m.created_at DESC
            LIMIT {limit}
            """
            results = db.query(cypher, params) if params else db.query(cypher)

        return results

    def scan_and_log(self) -> List[Dict[str, Any]]:
        results = self.compute_sheaf_surprise_score(limit=5)
        for res in results:
            if res.get("surprise_score", 0) > 0.8:
                logger.info(f"Sheaf Monitor Alert: High surprise edge detected: {res}")
        return results

    async def check_axiomatic_consistency(
        self,
        proposed_code: str,
        domain_context: Optional[str] = None,
        task_tags: Optional[List[str]] = None,
        repl_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        [CAG Pivot] Validates the proposed code against executable Axioms.
        Uses CLEAN PythonREPL without simulated mocks.
        """
        from ..mcp_integration.skills import get_axioms_manager

        axioms_mgr = get_axioms_manager()
        all_axioms = axioms_mgr.list_axioms()

        # 1. Filter for relevant Axioms
        axioms = []
        for name, axiom_meta in all_axioms.items():
            axiom_tags = axiom_meta.get("tags", [])
            is_relevant = False

            if task_tags and any(tag in axiom_tags for tag in task_tags):
                is_relevant = True
            elif "general" in axiom_tags:
                is_relevant = True
            elif not task_tags and not axiom_tags:
                is_relevant = True

            if is_relevant:
                axioms.append(name)

        if not axioms:
            logger.info(
                f"🛡️ Axiomatic Check: No specific guardrails for tags {task_tags}."
            )
            return {"status": "HEALTHY", "energy": 0.0, "axioms_run": []}

        logger.info(f"🛡️ Axiomatic Check: Running {len(axioms)} relevant guardrails...")

        # 2. Spawn temporary sandbox REPL (Clean Environment)
        repl = PythonREPL()

        # 3. Inject allowed utilities (No Mocks)
        import asyncio

        repl.namespace.update(
            {
                "asyncio": asyncio,
                "np": np,
            }
        )

        violations = []
        try:
            # 4. Run Proposed Code in Sandbox
            # Unpacking 4 values: stdout, stderr, result, is_err
            stdout, stderr, result, is_err = await repl.execute(proposed_code)

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
                        match = re.search(r"def (validate_\w+)", axiom["code"])
                        func_name = match.group(1) if match else None

                    if not func_name:
                        continue

                    # Execute validator against result or state
                    # If the code (e.g. assignments) returns None, we pass the local variables (state)
                    # so the validator can check side-effects (e.g. x=5).
                    if result is not None:
                        # Bind result to temporary variable to avoid massive f-string injection
                        repl.namespace["_axiom_target"] = result
                        val_call = f"{func_name}(_axiom_target)"
                    else:
                        # Pass a copy of locals (excluding internal vars)
                        repl.namespace["_axiom_target"] = {
                            k: v
                            for k, v in repl.namespace.items()
                            if not k.startswith("_") and k not in ["asyncio", "np"]
                        }
                        val_call = f"{func_name}(_axiom_target)"

                    _, val_err, val_res, _ = await repl.execute(val_call, silent=True)

                    if val_res is False:
                        violations.append(
                            f"Axiom '{axiom_name}' violated: {axiom.get('description', 'No description')}"
                        )
                except Exception as e:
                    logger.warning(f"Error processing axiom {axiom_name}: {e}")

            if violations:
                logger.warning(f"🚫 Axiomatic Violation detected: {violations}")
                return {
                    "status": "AXIOMATIC_VIOLATION",
                    "energy": 1.0,
                    "critique": "\n".join(violations),
                    "details": violations,
                    "axioms_run": axioms,
                    "should_halt": False,  # Changed to False to prioritize Self-Healing
                }

            return {
                "status": "HEALTHY",
                "energy": 0.0,
                "critique": None,
                "axioms_run": axioms,
                "should_halt": False,
            }
        except Exception as e:
            logger.error(f"Axiomatic validation system failure: {e}")
            return {"status": "HEALTHY", "energy": 0.0, "error": str(e)}


sheaf = SheafMonitor()
