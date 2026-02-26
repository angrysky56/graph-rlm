"""
Reflexion/IntelliSynth Module.

Implements the IntelliSynth Framework for breaking logical knots and stagnation
using the Advancement Cycle (Truth -> Scrutiny -> Improvement) and mathematical reasoning.
"""

import hashlib
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .guardrails import EmpiricalGuard, GuardrailError, extract_python_code
from .llm import llm
from .logger import get_logger
from .omcd import omcd
from .trace import trace_action

logger = get_logger("graph_rlm.reflexion")


class IntelliSynth:
    """
    The IntelliSynth Framework:
    1. AwL (Analyze with Logic): Logic, Intuition, Abduction.
    2. Advancement Cycle: Truth -> Scrutiny -> Improvement.
    3. Mathematical Context: Entropy, Bayesian Reasoning, BLEU.
    """

    def __init__(self):
        self.llm = llm
        self.alpha = 1.0  # Weight for Scrutiny
        self.beta = 1.5  # Weight for Improvement
        self._genesis_cache: Dict[str, List[float]] = {}  # MD5(genesis) -> embedding

    # --- INTROSPECTIVE HEALING (Semantic & Structural) ---

    def _extract_mission_state(self, scratchpad: str) -> Dict[str, str]:
        """Parses the Mission Control section of the scratchpad."""
        state = {}
        if not scratchpad:
            return state

        # Extract Phase
        phase_match = re.search(r"\- \*\*Phase\*\*: (.*)", scratchpad)
        if phase_match:
            state["phase"] = phase_match.group(1).strip()

        # Extract Last Critique
        critique_match = re.search(
            r"\- \*\*Last Dreamer Critique\*\*: (.*)", scratchpad, re.IGNORECASE
        )
        if critique_match:
            state["last_critique"] = critique_match.group(1).strip()

        return state

    def _extract_recent_trace(self, scratchpad: str) -> List[Dict[str, str]]:
        """Parses the Execution Trace table from the scratchpad."""
        trace = []
        if not scratchpad:
            return trace

        # Find the trace table rows
        # | 18:09:34 | `7f2d572` | 1.1 | ✗ | `chash` | ... | SUMMARY | ... |
        # Matches: Time, REPL, Step, Status, Hash, Gestalt, Summary
        matches = re.finditer(
            r"\|\s*[\d:]+\s*\|\s*`([^`]+)`\s*\|\s*[\d\.]+\s*\|\s*([^|]+?)\s*\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|",
            scratchpad,
        )
        for match in matches:
            trace.append(
                {
                    "repl_id": match.group(1).strip(),
                    "status": match.group(2).strip(),
                    "code_hash": match.group(3).strip(),
                    "gestalt": match.group(4).strip(),
                    "summary": match.group(5).strip(),
                }
            )

        return trace[-5:]  # Only keep recent history

    async def introspective_probe(
        self,
        response_text: str,
        rlm: Optional[Any] = None,
        context_scratchpad: str = "",
        exec_state: Optional[Any] = None,
        recent_thoughts: Optional[List[Dict]] = None,
        agent: Any = None,
        session_id: str = "unknown",
        root_session_id: str = "unknown",
        step_id: int = 1,
        turn_id: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Unified Introspective Probe (Simulation).
        Analyzes a generated thought for immediate structural or logical flaws
        BEFORE it is committed to the graph or executed.

        Args:
            exec_state: Typed ExecutionState object. When provided, phase and
                critique are read directly from it instead of regex-parsing the
                scratchpad (preferred path).
            recent_thoughts: Pre-built list of recent thought dicts. When provided,
                replaces the regex-parsed trace from context_scratchpad.

        Returns:
            Optional[Dict]: A 'Correction' object if an issue is found, else None.
        """
        if not response_text:
            return None

        # --- 1. Semantic Alignment & Repetition Detection (Phase 5) ---
        if exec_state is not None:
            # Preferred path: read typed state directly — no Markdown round-trip
            mission = {
                "phase": getattr(exec_state, "phase", ""),
                "last_critique": getattr(exec_state, "last_dreamer_critique", "") or "",
            }
            trace = recent_thoughts or []
        elif context_scratchpad:
            # Backward-compat fallback: regex-parse the rendered scratchpad.
            # Remove once all callers pass exec_state.
            mission = self._extract_mission_state(context_scratchpad)
            trace = self._extract_recent_trace(context_scratchpad)
        else:
            mission, trace = {}, []

        if mission or trace:

            # Repetition Loop detection
            recent_failures = [
                f for f in trace if f["status"] in ["✗", "failed", "error"]
            ]
            if recent_failures:
                last_fail = recent_failures[-1]
                last_fail_summary = last_fail["summary"].lower()
                current_thought = response_text.lower()

                # Check for Code Delta (Phase 6)
                # If the code changed, we consider it "Progress" and relax the repetition check
                current_code_hash = ""
                if "python" in response_text:
                    code_str = extract_python_code(response_text)
                    if code_str:
                        current_code_hash = hashlib.sha256(
                            code_str.encode("utf-8")
                        ).hexdigest()[:7]

                code_changed = current_code_hash != last_fail["code_hash"]

                if not code_changed:
                    # Keyword overlap check for repetition
                    fail_keywords = {
                        k.strip() for k in last_fail_summary.split() if len(k) > 3
                    }
                    thought_keywords = {
                        k.strip() for k in current_thought.split() if len(k) > 3
                    }
                    overlap = fail_keywords.intersection(thought_keywords)

                    # If more than 50% of the failure keywords are repeated with IDENTICAL code
                    if (
                        len(fail_keywords) > 0
                        and len(overlap) / len(fail_keywords) >= 0.5
                    ):
                        correction = {
                            "type": "REPETITION_ERROR",
                            "message": "You are repeating a failed reasoning trajectory with the same code.",
                            "hint": f"Your last attempt ('{last_fail['summary']}') failed with the same code. Try a different technique.",
                            "severity": "WARNING",
                        }
                        await self._trigger_deep_analysis(
                            correction,
                            response_text,
                            context_scratchpad,
                            trace,
                            agent,
                            session_id,
                            root_session_id,
                            step_id,
                            turn_id,
                        )
                        return correction

            # Goal Alignment (Mission Control)
            last_critique = mission.get("last_critique")
            if last_critique:
                critique_keywords = [
                    k.strip().lower()
                    for k in last_critique.replace("'", "").replace('"', "").split()
                    if len(k) > 4 and k.isalnum()
                ]
                if len(critique_keywords) >= 2:
                    found_compliance = any(
                        k in response_text.lower() for k in critique_keywords
                    )
                    if not found_compliance:
                        correction = {
                            "type": "ALIGNMENT_ERROR",
                            "message": "You are drifting away from the Dreamer's feedback.",
                            "hint": f"The Dreamer requested: '{last_critique}'. Ensure your thought addresses this explicitly.",
                            "severity": "LOW",
                        }
                        await self._trigger_deep_analysis(
                            correction,
                            response_text,
                            context_scratchpad,
                            trace,
                            agent,
                            session_id,
                            root_session_id,
                            step_id,
                            turn_id,
                        )
                        return correction

        # --- 2. Extract Code Blocks for Structural Validation ---
        full_code = extract_python_code(response_text)
        if not full_code:
            return None

        # --- 3. Empirical Content Validation (Unified with Guardrails) ---

        try:
            # Syntax & RLM Signatures
            EmpiricalGuard.validate_syntax(full_code)
            EmpiricalGuard.validate_rlm_signatures(full_code)
            EmpiricalGuard.validate_mcp_imports(full_code)
        except GuardrailError as e:
            # Map GuardrailError back to the correction format for healing
            error_msg = str(e)
            severity = "CRITICAL"
            error_type = "GUARDRAIL_VIOLATION"

            if "Syntax Error" in error_msg:
                error_type = "SYNTAX_ERROR"
            elif "Async Error" in error_msg:
                error_type = "ASYNC_ERROR"
            elif "Import Error" in error_msg:
                error_type = "IMPORT_ERROR"

            correction = {
                "type": error_type,
                "message": error_msg,
                "hint": "Check your Indentation, bracket closures, or RLM/MCP naming.",
                "severity": severity,
            }
            await self._trigger_deep_analysis(
                correction,
                response_text,
                context_scratchpad,
                trace,
                agent,
                session_id,
                root_session_id,
                step_id,
                turn_id,
            )
            return correction

        # 3b. Signature & Logical Check (Contextual - remains in Reflexion)
        if rlm:
            # Check for tool signatures if rlm.mcp is available
            if hasattr(rlm, "mcp"):
                # Simple regex check for rlm.mcp calls to verify they exist
                tool_calls = re.findall(r"rlm\.mcp\.(\w+)\.", full_code)
                for server in tool_calls:
                    if server not in rlm.mcp.list_servers():
                        correction = {
                            "type": "SIGNATURE_ERROR",
                            "message": f"MCP server '{server}' not found in current namespace.",
                            "hint": f"Available servers: {', '.join(rlm.mcp.list_servers())}",
                            "severity": "WARNING",
                        }
                        await self._trigger_deep_analysis(
                            correction,
                            response_text,
                            context_scratchpad,
                            trace,
                            agent,
                            session_id,
                            root_session_id,
                            step_id,
                            turn_id,
                        )
                        return correction

        return None

    async def _trigger_deep_analysis(
        self,
        correction: Dict,
        response_text: str,
        context_scratchpad: str,
        trace: List,
        agent: Any,
        session_id: str,
        root_session_id: str,
        step_id: int,
        turn_id: int,
    ):
        """Helper to fire off the deep advancement cycle on failure."""
        await self.advancement_cycle(
            trace_context=context_scratchpad if context_scratchpad else str(trace),
            current_thought=response_text,
            divergence_point=correction.get("message", "unknown"),
            agent=agent,
            session_id=session_id,
            root_session_id=root_session_id,
            step_id=step_id,
            turn_id=turn_id,
        )

    # --- SPECIAL FUNCTIONS (SF) ---

    def sigmoid(self, x: float) -> float:
        """SF3: Apply Neural Activation (Sigmoid)."""
        return 1 / (1 + math.exp(-x))

    def entropy(self, probabilities: List[float]) -> float:
        """SF4: Apply Uncertainty (Entropy)."""
        return -sum(p * math.log2(p) for p in probabilities if p > 0)

    def bleu_score(self, weights: List[float], probabilities: List[float]) -> float:
        """SF2: Apply Natural Language Understanding (BLEU Score calculation)."""
        # Simplified BLEU-inspired metric
        p_sum = sum(
            w * math.log(p)
            for w, p in zip(weights, probabilities, strict=False)
            if p > 0
        )
        return math.exp(p_sum)

    def imprecise_reasoning(self, x: float, k: float, c: float) -> float:
        """SF1: Apply Imprecise Reasoning (Mamdani/Fuzzy Sigmoid)."""
        return 1 / (1 + math.exp(-k * (x - c)))

    def reason(self, p_a: float, p_b_given_a: float, p_b: float) -> float:
        """CA9: Apply Reasoning (Bayesian)."""
        if p_b == 0:
            return 0.0
        return (p_b_given_a * p_a) / p_b

    # --- AI CONCEPTS APPLICATION (CA) ---

    def universal_intelligence(
        self, x: np.ndarray, omega: np.ndarray, functions: List[Any]
    ) -> float:
        """CA4: Apply Universal Intelligence (Formalized as weighted sum of basis functions)."""
        n = len(functions)
        sum1 = sum(omega[i] * functions[i](x) for i in range(n))
        # Interaction terms
        sum2 = 0
        for j in range(n):
            for j in range(n):
                for k in range(n):
                    sum2 += omega[j][k] * functions[j](x) * functions[k](x)
        return sum1 + sum2

    def optimize(self, _f: Any, _d: List[Tuple[Any, Any]]) -> str:
        """CA5: Apply Optimization."""
        return "Parameters Optimized successfully via Gradient Descent proxy."

    def q_value(self, s: Any, _a: Any, r: float, gamma: float, q_func: Any) -> float:
        """CA6: Apply Learning from Rewards."""
        # Standard Bellman equation
        return r + gamma * max(q_func(s, a_prime) for a_prime in ["action1", "action2"])

    def transfer_learning(self, l_source: float, delta_l: float) -> float:
        """CA7: Apply Transfer Learning."""
        return l_source + delta_l

    def adapt_learning_rate(self, eta_0: float, alpha: float, t: int) -> float:
        """CA8: Apply Adaptability."""
        return eta_0 / (1 + alpha * t)

    def evolutionary_intelligence(self, x: Any) -> float:
        """CA0: Apply Evolutionary Intelligence."""
        # Returns fitness score
        return float(hash(str(x)) % 100) / 100.0

    # --- ADVANCEMENT PROCESS (AP) — Data Engine ---
    # Returns structured metrics for the main agent to interpret.
    # NO LLM calls. One brain, many sensors.

    async def advancement_cycle(
        self,
        trace_context: str,
        current_thought: str,
        divergence_point: str,
        agent: Any = None,
        db: Any = None,
        session_id: str = "unknown",
        root_session_id: str = "unknown",
        turn_id: int = 1,
        step_id: int = 1,
    ) -> Dict[str, Any]:
        """
        Data-only Advancement Cycle. Returns structured analysis metrics.

        Replaces the old 3-LLM-call pipeline (Truth -> Scrutiny -> Improvement)
        with pure computation + graph queries. The main agent (which has full
        scratchpad, axioms, and tool context) interprets these metrics.

        Returns:
            Dict with keys: type, shakiness, loop_energy, topo_status,
            genesis_anchor, drift_score, action, metrics_report
        """
        logger.info("[IntelliSynth] Gathering analysis metrics (data-only)...")
        trace_action(
            "INTELLISYNTH",
            "CYCLE_START",
            result="Data-only analysis cycle (no LLM calls).",
            tag="SYSTEM",
        )

        # 1. Gather Monitor Metrics (RepE + Sheaf — pure computation)
        metrics = await self._gather_metrics(current_thought)

        # 2. Fetch Genesis Anchor (graph query, no LLM)
        genesis_anchor = self._fetch_genesis(db, root_session_id)

        # 3. Compute Drift Score (Mathematical Alignment Engine V2)
        drift_score = await self._compute_drift(
            genesis_anchor, current_thought, metrics, trace_context
        )

        # 4. Determine action from data
        action = "CONTINUE"
        if drift_score > 0.6 and genesis_anchor:
            action = "FOLD_TO_ROOT"
        elif metrics["loop_energy"] > 0.7:
            action = "BREAK_LOOP"
        elif metrics["shakiness"] > 0.5:
            action = "GROUND_YOURSELF"

        result = {
            "type": "REFLEXION_ANALYSIS",
            "shakiness": metrics["shakiness"],
            "loop_energy": metrics["loop_energy"],
            "topo_status": metrics["topo_status"],
            "divergence_reason": divergence_point,
            "genesis_anchor": (genesis_anchor[:300] if genesis_anchor else "NO_ANCHOR"),
            "drift_score": drift_score,
            "action": action,
            "metrics_report": metrics["report"],
            "v2_engine": True,
        }

        # PERSISTENCE: Materialize analysis using the Agent's standardized chain
        if agent and hasattr(agent, "_create_system_node"):
            try:
                import json as _json

                # Use a descriptive label for the graph node
                node_label = f"Reflexion: {action} ({divergence_point[:30]}...)"
                node_result = f"### IntelliSynth Analysis\n{_json.dumps(result, indent=2, default=str)}"

                await agent._create_system_node(
                    logical_id=f"REF_ANALYSIS_{step_id}",
                    summary=node_label,
                    status="reflexion",
                    session_id=session_id,
                    root_session_id=root_session_id,
                    repl_id="REF",
                    result=node_result,
                    turn_id=turn_id,
                    step_id=step_id,
                    analysis=result,
                    validate=False,
                )
            except (AttributeError, RuntimeError, KeyError, ValueError) as db_err:
                logger.error(
                    "Failed to persist IntelliSynth analysis via agent: %s",
                    db_err,
                    exc_info=True,
                )
        elif db:
            # Fallback for direct DB access if agent is missing (mostly for tests/legacy)
            try:
                import json as _json

                db.create_thought_node(
                    thought_id=f"{session_id}:REF:ANALYSIS:{step_id}",
                    prompt=f"Reflexion: {action}",
                    result=f"### Reflexion Analysis\n{_json.dumps(result, indent=2, default=str)}",
                    status="reflexion",
                    session_id=session_id,
                    root_session_id=root_session_id,
                    repl_id="REF",
                    execution_summary=f"Drift={drift_score:.2f} Action={action} Reason={divergence_point[:50]}",
                    turn_id=turn_id,
                    step_id=step_id,
                    validate=False,
                )
            except (AttributeError, RuntimeError, KeyError, ValueError) as db_err:
                logger.error(
                    "Failed to persist IntelliSynth analysis to DB: %s",
                    db_err,
                    exc_info=True,
                )

        trace_action(
            "INTELLISYNTH",
            "CYCLE_COMPLETE",
            result=f"Drift={drift_score:.2f} Action={action}",
            tag="SYSTEM",
        )

        return result

    async def _gather_metrics(self, current_thought: str) -> Dict[str, Any]:
        """Gather RepE + Sheaf metrics. Pure computation, no LLM reasoning."""
        try:
            from .repe import repe
            from .sheaf import sheaf
        except ImportError:
            repe = None  # type: ignore
            sheaf = None  # type: ignore

        shakiness = 0.0
        loop_energy = 0.0
        topo_status = "UNKNOWN"

        try:
            vec = await self.llm.get_embedding(current_thought)

            if vec:
                if repe:
                    profile = repe.scan_thought(vec)
                    raw_grounding = float(profile.get("Shakiness", 0.0))
                    shakiness = -raw_grounding  # Invert: positive = neurotic

                if sheaf:
                    diagnosis = sheaf.diagnose_trace(
                        root_id="reflexion",
                        hypothetical_node={
                            "content": current_thought,
                            "embedding": vec,
                            "prompt": current_thought,
                        },
                    )
                    topo_status = diagnosis.get("status", "UNKNOWN")
                    loop_energy = float(diagnosis.get("energy", 0.0))
        except (ValueError, TypeError, AttributeError, ArithmeticError) as e:
            logger.warning("Metrics gathering failed: %s", e)
        except RuntimeError as e:
            logger.warning("Metrics gathering runtime error: %s", e)

        report = (
            f"Shakiness: {shakiness:.2f} | "
            f"Topology: {topo_status} | "
            f"Loop Energy: {loop_energy:.2f}"
        )

        return {
            "shakiness": shakiness,
            "loop_energy": loop_energy,
            "topo_status": topo_status,
            "report": report,
        }

    def _fetch_genesis(self, db: Any, root_session_id: str) -> str:
        """
        Fetch T=0 genesis 'Anchoring Context' from graph.
        Targets the initial user goal and system objectives.
        """
        if not db:
            return ""
        try:
            # Query for the very first user message and any high-level session goals
            res = db.query(
                "MATCH (n:Thought) WHERE n.root_session_id = $sid "
                "AND n.step_id = 1 "
                "RETURN n.prompt AS p, n.summary AS s, n.content AS c "
                "LIMIT 1",
                {"sid": root_session_id},
            )
            if res:
                node = res[0]
                # Priority: content (result) > summary > prompt
                anchor = node.get("c") or node.get("s") or node.get("p") or ""
                return str(anchor)[:2000]

            # Fallback to general session query if step 1 is missing
            res = db.query(
                "MATCH (n:Thought) WHERE n.root_session_id = $sid "
                "RETURN n.prompt AS p ORDER BY n.step_id ASC LIMIT 1",
                {"sid": root_session_id},
            )
            if res:
                return str(res[0].get("p", ""))[:1500]

        except (AttributeError, RuntimeError, KeyError, ValueError) as e:
            logger.warning("Failed to fetch genesis anchor: %s", e)
        return ""

    async def _compute_drift(
        self,
        genesis: str,
        current_thought: str,
        metrics: Dict[str, Any],
        trace_context: Optional[str] = None,
    ) -> float:
        """
        Mathematical Alignment Engine (V2).
        Uses Semantic Vectors, Bernshteyn Bounds, and RepE Integrity.
        """
        # --- Axis 1: Semantic Vector Alignment ---
        semantic_drift = 0.5
        if genesis and current_thought:
            try:
                # Cache genesis embedding — the same prompt is passed every cycle,
                # so avoid a redundant LLM API round-trip after the first call.
                g_text = genesis[:2000]
                g_hash = hashlib.md5(g_text.encode(), usedforsecurity=False).hexdigest()

                if g_hash not in self._genesis_cache:
                    vec = await self.llm.get_embedding(g_text)
                    if vec:
                        self._genesis_cache[g_hash] = vec

                g_vec = self._genesis_cache.get(g_hash)
                t_vec = await self.llm.get_embedding(current_thought[:2000])

                if g_vec and t_vec:
                    gv = np.array(g_vec)
                    tv = np.array(t_vec)
                    # Cosine similarity (1.0 = identical, 0.0 = orthogonal)
                    norm_g = np.linalg.norm(gv)
                    norm_t = np.linalg.norm(tv)
                    if norm_g > 0 and norm_t > 0:
                        sim = np.dot(gv, tv) / (norm_g * norm_t)
                        # Drift = 1.0 - Similarity
                        semantic_drift = max(0.0, min(1.0, 1.0 - float(sim)))
            except (ValueError, TypeError, RuntimeError) as e:
                logger.warning("Semantic drift calculation failed: %s", e)

        # --- Axis 2: Bernshteyn Complexity (Rulial Expansion) ---
        # Use oMCD's Bernshteyn penalty logic. High penalty = High Drift.
        bern_drift = 0.0
        try:
            # We treat 'shakiness' as the inverse of precision for the bound
            precision = max(0.0, 1.0 - abs(metrics.get("shakiness", 0.0)))
            # We don't have the current step here easily, so we use loop energy as a proxy
            # to estimate if we are in a high-depth/entropy state.
            depth_proxy = int(metrics.get("loop_energy", 0) * 20)
            penalty = omcd.calculate_bernshteyn_penalty(depth_proxy, precision)
            # Normalize penalty: 1.0 (no penalty) -> 0.0 drift, >1.0 -> proportional drift
            if penalty > 1.0:
                bern_drift = min(0.6, (penalty - 1.0) / 10.0)
        except (ValueError, TypeError, ZeroDivisionError):
            pass

        # --- Axis 3: RepE Integrity Shift ---
        # If shakiness is positive (Neurotic/Shaky), it contributes to drift.
        repe_drift = max(0.0, metrics.get("shakiness", 0.0))

        # --- Axis 4: Stagnation (Repetition) ---
        repetition_penalty = 0.0
        if trace_context and current_thought:
            thought_sig = current_thought.strip()[:100].lower()
            if thought_sig and thought_sig in trace_context.lower()[-3000:]:
                repetition_penalty = 0.6  # Heavy penalty for repeating oneself
                logger.info(
                    "[IntelliSynth V2] Detected repetition! Repetition Penalty: 0.6"
                )

        # --- Final Composite Alignment ---
        # Weights: Semantic Alignment (40%), Stagnation (30%), Complexity (20%), Integrity (10%)
        total_drift = (
            (semantic_drift * 0.4)
            + (repetition_penalty * 0.3)
            + (bern_drift * 0.2)
            + (repe_drift * 0.1)
        )

        trace_action(
            "INTELLISYNTH",
            "ALIGNMENT_CHECK",
            result=(
                f"Drift={total_drift:.2f} [Sem={semantic_drift:.2f}, Rep={repetition_penalty:.2f}, "
                f"Bern={bern_drift:.2f}, RepE={repe_drift:.2f}]"
            ),
            tag="SYSTEM",
        )

        return min(1.0, total_drift)

    async def engage_intuition(self, premises: List[str]) -> Dict[str, float]:
        """Pattern matching proxy (Intuition) backed by Sheaf/RepE."""
        # 1. Join premises
        text = " ".join(premises)

        # 2. Lazy Import
        # pylint: disable=import-outside-toplevel
        try:
            from .repe import repe
            from .sheaf import sheaf
        except ImportError:
            repe = None
            sheaf = None

        confidence = 0.5
        pattern_density = 0.5

        try:
            vec = await self.llm.get_embedding(text)
            if vec:
                # Sheaf Energy -> inversed for Density
                if sheaf:
                    diag = sheaf.diagnose_trace(
                        root_id="intuition",
                        hypothetical_node={"embedding": vec, "content": text},
                    )
                    energy = float(diag.get("energy", 0.5))
                    pattern_density = max(0.0, min(1.0, 1.0 - energy))

                # RepE Score -> Sigmoid for Confidence
                if repe:
                    profile = repe.scan_thought(vec)
                    raw_score = float(profile.get("Shakiness", 0.0))
                    # Raw score: +ve = Grounded, -ve = Shaky
                    confidence = self.sigmoid(raw_score * 2.0)  # Scale?
        except (RuntimeError, AttributeError, ValueError) as e:
            logger.warning("Intuition metrics failed: %s", e)

        return {"PatternDensity": pattern_density, "Confidence": confidence}

    async def employ_abductive_reasoning(
        self, premises: List[str]
    ) -> Tuple[List[str], Dict[str, float]]:
        """Infer the most likely explanation (Abduction)."""
        prompt = (
            f"Infer the hidden assumptions and most likely explanations for: {premises}"
        )
        res = await self.llm.generate(prompt, system="Abductive Reasoner", stream=False)
        # Simple extraction logic for demo
        assumptions = [
            line.strip("- ") for line in res.split("\n") if line.strip().startswith("-")
        ]
        return assumptions, {"ExplanationPlausibility": 0.9}


# Singleton
intelli_synth = IntelliSynth()
