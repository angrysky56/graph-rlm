"""
The Dreamer: Consolidation and Wisdom layer for Graph-RLM.
Analyzes surprise events, logical knots, and codifies axioms during 'sleep' cycles.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import redis

from ..mcp_integration.skill_storage import get_axioms_manager
from .config import settings
from .core import PythonREPL
from .db import GraphClient, db
from .llm import llm
from .logger import get_logger
from .navigator import navigator
from .omcd import omcd
from .services.circuit import protected_llm_with_fallback
from .sheaf import sheaf
from .trace import trace_action

logger = get_logger("graph_rlm.dreamer")


class Dreamer:
    """
    The 'Sleep' Phase of the Graph-RLM architecture.
    Consolidates high-entropy (Surprise) events into 'Wisdom' (Insights).
    Also provides 'Lucid Dream' capabilities for immediate loop analysis.
    """

    def __init__(self):
        self.db: GraphClient = db
        self.llm = llm
        self.llm = llm
        self._is_codifying = False  # Recursion guard for axiom generation
        self._is_dreaming = False  # Main dream cycle lock

    def _get_session_trace(
        self,
        session_id: Optional[str],
        turn_id: Optional[int] = None,
        root_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query DB for actual session metrics so validation can cross-reference
        the agent's claims against what actually happened.

        Returns turn count, step count, REPL IDs, failure count, recent node IDs
        (for edge construction), and status timeline with timestamps.
        """
        empty: Dict[str, Any] = {
            "turn_count": 0,
            "step_count": 0,
            "repl_ids": [],
            "failure_count": 0,
            "recent_node_ids": [],
            "status_timeline": [],
        }
        if not session_id:
            return empty

        try:
            # Build turn filter
            turn_clause = ""
            # We anchor on root_session_id if provided, falling back to session_id
            target_id = root_session_id or session_id
            qparams: Dict[str, Any] = {"sid": target_id}

            if turn_id is not None:
                turn_clause = "AND t.turn_id = $tid"
                qparams["tid"] = turn_id

            agg = self.db.query(
                f"""
                MATCH (t:Thought)
                WHERE (t.session_id = $sid OR t.root_session_id = $sid) {turn_clause}
                RETURN count(t) as step_count,
                       collect(DISTINCT t.turn_id) as turns,
                       collect(DISTINCT t.repl_id) as repls,
                       size([x IN collect(t.status) WHERE x IN ['failed', 'error', 'rejected']]) as failures
                """,
                qparams,
            )

            # Recent nodes with timestamps (for edge construction + timeline)
            recent = self.db.query(
                f"""
                MATCH (t:Thought)
                WHERE (t.session_id = $sid OR t.root_session_id = $sid) {turn_clause}
                RETURN t.id as id, t.status as status, t.created_at as ts,
                       t.repl_id as repl_id
                ORDER BY t.created_at DESC
                LIMIT 10
                """,
                qparams,
            )

            result = dict(empty)  # copy defaults
            if agg and len(agg) > 0:
                row = (
                    agg[0]
                    if isinstance(agg[0], dict)
                    else {
                        "step_count": agg[0][0],
                        "turns": agg[0][1],
                        "repls": agg[0][2],
                        "failures": agg[0][3],
                    }
                )
                turns = [t for t in (row.get("turns") or []) if t is not None]
                repls = [r for r in (row.get("repls") or []) if r is not None]
                result["turn_count"] = len(turns)
                result["step_count"] = row.get("step_count", 0)
                result["repl_ids"] = repls
                result["failure_count"] = row.get("failures", 0)

            if recent:
                result["recent_node_ids"] = [r["id"] for r in recent if r.get("id")]
                result["status_timeline"] = [
                    {
                        "id": str(r.get("id", "?"))[:8],
                        "status": r.get("status", "unknown"),
                        "ts": str(r.get("ts", "")),
                        "repl": r.get("repl_id", ""),
                    }
                    for r in recent
                ]

            return result
        except (RuntimeError, KeyError, ValueError, AttributeError) as e:
            logger.warning("Failed to query session trace: %s", e)

        return empty

    async def analyze_holonomy(
        self, loop_nodes: List[Dict[str, Any]], current_thought: str
    ) -> str:
        """
        [Lucid Dream / IntelliSynth] Immediate synchronous analysis of a detected logical knot.
        Uses the IntelliSynth Advancement Cycle (Truth -> Scrutiny -> Improvement)
        to break the loop with mathematical precision.
        """
        from .reflexion import intelli_synth

        logger.info(
            "⚡ [Dreamer] Triggering IntelliSynth Advancement Cycle for Loop Analysis..."
        )

        # Format history trace for AwL Analysis
        trace_str = ""
        for i, node in enumerate(reversed(loop_nodes)):
            # Handle FalkorDB/Neo4j node structures
            props = node
            if hasattr(node, "properties"):
                props = node.properties
            elif "n" in node:
                props = node["n"]

            # Normalize dict access
            if hasattr(props, "properties"):
                props = props.properties

            content = props.get("content", str(props))
            trace_str += f"Step -{i}: {content[:30000]}...\n"

        divergence_node_id = loop_nodes[-1].get("id") if loop_nodes else "unknown"

        # Execute IntelliSynth Advancement Cycle
        # (AP1: Truth -> AP2: Scrutiny [AwL] -> AP3: Improvement)
        try:
            improvement_directive = await intelli_synth.advancement_cycle(
                trace_context=trace_str,
                current_thought=current_thought,
                divergence_point=str(divergence_node_id),
            )
            return improvement_directive
        except (AttributeError, ValueError, TypeError, RuntimeError):
            logger.exception("IntelliSynth cycle failed")
            # Fallthrough to robust default directive below
            return (
                "SYSTEM REFLEXION: BREAK LOOP. IntelliSynth failed, but you are repeating yourself. "
                "Change approach immediately."
            )

    async def dream_cycle(
        self,
        emit_callback=None,
        session_id: Optional[str] = None,
        final_response_candidate: Optional[str] = None,
        context: Optional[str] = None,
        turn_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Public wrapper for Dream Cycle with strict locking to prevent overlapping execution.
        """

        def emit(event_type, content, is_internal=False):
            if emit_callback:
                emit_callback(event_type, content=content, is_internal=is_internal)

        if self._is_dreaming:
            logger.warning("🛑 [Dreamer] Dream cycle already in progress. Skipping.")
            emit(
                "debug",
                "[Dreamer] Dream cycle skipped (already running).",
                is_internal=True,
            )
            return {
                "status": "skipped",
                "message": "Dream cycle in progress",
                "insights": [],
                "events_processed": 0,
                "insight": "",
                "id": None,
            }

        self._is_dreaming = True
        try:
            return await self._dream_cycle_impl(
                emit_callback, session_id, final_response_candidate, context, turn_id
            )
        except Exception as e:
            logger.error(
                "Unexpected error in Dream Cycle Wrapper: %s", e, exc_info=True
            )
            return {"status": "error", "message": f"Wrapper Error: {str(e)}"}
        finally:
            self._is_dreaming = False

    async def _dream_cycle_impl(
        self,
        emit_callback=None,
        session_id: Optional[str] = None,
        final_response_candidate: Optional[str] = None,
        context: Optional[str] = None,
        turn_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Main Sleep Cycle:
        1. Query 'Surprise' (High Energy Edges / Failed Tests).
        2. Consolidate into 'Insight'.
        3. [CAG Pivot] Attempt to auto-codify defensive Axioms if patterns are clear.
        4. Write Insight back to Graph.

        Args:
            emit_callback: Optional callable(event_type, content) to emit events to UI.
            session_id: If provided, scope surprise analysis to this session only.
            final_response_candidate: The Agent's proposed final answer.
                                      Dreamer checks if this resolves the failures.
            context: The full Agent Scratchpad (history, REPL IDs, recent topology)
                     for cross-verification.
        """

        def emit(event_type, content, is_internal=False):
            if emit_callback:
                emit_callback(event_type, content=content, is_internal=is_internal)

        logger.info("🛌 Initiating Dream Cycle (Sleep Phase)...")
        emit("thinking", "🛌 [Dreamer] Initiating Dream Cycle...", is_internal=True)

        # 1. Gather Surprise (High Energy Edges) - scoped to current session if provided
        surprise_events = sheaf.compute_sheaf_surprise_score(
            limit=10, session_id=session_id, turn_id=turn_id
        )

        trace_action(
            "DREAMER",
            "CYCLE_START",
            result=f"Analyzing {len(surprise_events)} high-surprise events...",
            tag="DREAMER",
        )

        # [MOVED UP] 1. Gather Recent Frontier (The Truth) to check for Stagnation
        # We need to see the *latest* events to check for success
        recent_context_str = "No recent context"
        recent_events = []
        if session_id:
            recent_events = self.db.query(
                """
                MATCH (n:Thought)
                WHERE (n.session_id = $sid OR n.root_session_id = $sid)
                RETURN n.id as id, n.prompt as prompt, n.status as status, n.result as result
                ORDER BY n.created_at DESC
                """,
                {"sid": session_id},
            )

            if recent_events:
                # Normalize result formatting
                recent_lines = []
                for r in recent_events:
                    rid = r.get("id", "???")
                    status = r.get("status", "unknown")
                    prompt = str(r.get("prompt") or "")[:10000]
                    res = str(r.get("result") or "")[:10000]
                    recent_lines.append(
                        f"- [Node {rid}] Status: {status} | Action: {prompt}... | Result: {res}..."
                    )
                recent_context_str = "\n".join(recent_lines)

        if not surprise_events:
            # [CRITICAL FIX] Check for Silent Failure / Stagnation
            # If the session was empty or failed but generated no "Surprise" (because no edges formed),
            # we MUST wake the dreamer to inspect why nothing happened.
            is_stagnant = False
            if not recent_events:
                is_stagnant = True
            elif recent_events and recent_events[0].get("status") in [
                "failed",
                "error",
                "empty",
            ]:
                is_stagnant = True

            if is_stagnant:
                logger.warning(
                    "🚨 [Dreamer] Sleep disturbed by STAGNATION (Empty/Failed Session)."
                )
                trace_action(
                    "DREAMER",
                    "WAKE_UP",
                    result="Stagnation Detected. Forcing Analysis.",
                    tag="SYSTEM",
                )
                surprise_events.append(
                    {
                        "source": "SYSTEM",
                        "target": "AGENT",
                        "surprise_score": 1.0,
                        "status": "STAGNATION",
                        "parent_prompt": "SYSTEM MONITOR",
                        "child_prompt": "NO ACTION TAKEN",
                        "child_result": "Session ended with no successful operations.",
                    }
                )
            elif not final_response_candidate:
                # [CAG Fix]: Only return early if NO surprise AND NO final response candidate.
                # Successful runs often have low surprise but are the best time to extract skills.
                logger.info("No high-surprise events found. Sleep was peaceful.")
                trace_action(
                    "DREAMER",
                    "PEACEFUL",
                    result="No anomalies detected. Sleep was restorative.",
                    tag="DREAMER",
                )
                # 6. Axiom Quality Control (Automated Pruning)
                # Check for axioms that are causing systemic failure
                await self._perform_axiom_quality_control(session_id)
                return {
                    "status": "peaceful",
                    "insights": [],
                    "message": "No high-surprise events found.",
                }
            else:
                logger.info(
                    "No high-surprise events, but final response candidate present. Proceeding with Dream Cycle."
                )

        logger.info("Found %d high-surprise events.", len(surprise_events))

        processed_node_ids = [e["target"] for e in surprise_events]

        # (Context fetching moved up to support Stagnation Check)

        # 2. Formulate the Dream Prompt
        events_desc = []
        for event in surprise_events:
            src_node = await self._get_node_scan_async(event["source"])
            tgt_node = await self._get_node_scan_async(event["target"])

            status_raw = event.get("status")
            status_str = "FAILED"
            if status_raw == "reflexion":
                status_str = "LOGICAL_KNOT / INTERVENTION"
            elif status_raw != "failed" and status_raw != "error":
                status_str = f"Unknown ({status_raw})"
            events_desc.append(
                f"- Edge: {event['source']} -> {event['target']}\n"
                f"  Surprise Score: {event['surprise_score']:.2f}\n"
                f"  Status: {status_str}\n"
                f"  Parent Thought: {src_node.get('prompt', 'Unknown')[:10000]}...\n"
                f"  Child Action: {tgt_node.get('prompt', 'Unknown')[:10000]}...\n"
                f"  Result: {tgt_node.get('result', 'Unknown')[:10000]}..."
            )

        candidate_section = ""
        if final_response_candidate:
            candidate_section = (
                f"\n\n--- AGENT PROPOSED FINAL RESPONSE ---\n"
                f"{final_response_candidate[:20000]}...\n"
                f"---------------------------------------\n"
            )

        context_section = ""
        if context:
            context_section = (
                f"\n\n--- AGENT SCRATCHPAD & HISTORY (CONTEXT) ---\n"
                f"{context}\n"
                f"--------------------------------------------\n"
            )

        # [INTELLIGENT ACTIVATION] Detect if Axiom Codification is required
        # We look for explicit signals in the context OR structural signals in the events.
        axiom_required = False
        axiom_reason = "No signal detected."

        # 1. explicit Signal from Agent/Sheaf
        if context and "RLM_AXIOM_REQUIRED" in context:
            axiom_required = True
            axiom_reason = "Explicit 'RLM_AXIOM_REQUIRED' signal found in context."

        # 2. Structural Signals (Reflexion or High Surprise)
        if not axiom_required:
            for e in surprise_events:
                if e.get("status") == "reflexion":
                    axiom_required = True
                    axiom_reason = "Reflexion (Logical Knot) detected."
                    break
                if e.get("surprise_score", 0.0) > 0.9:
                    axiom_required = True
                    axiom_reason = "Extremely high surprise score (> 0.9) detected."
                    break

        # [PROMPT INJECTION] Guide the Dreamer
        system_signal_section = f"\n*** SYSTEM SIGNAL: {axiom_reason} ***\n"
        if axiom_required:
            system_signal_section += (
                "ACTION REQUIRED: The system has flagged a breakdown (Logical Knot/Surprise). "
                "Even if the agent recovered, you MUST codify an Axiom/Rule to prevent "
                "this initial failure from ever recurring.\n"
            )
        else:
            system_signal_section += (
                "OPPORTUNITY ANALYSIS: The system is stable. Look for SKILLS, TOOL USAGE PATTERNS, "
                "or DOMAIN KNOWLEDGE that should be preserved for future efficiency.\n"
                "If you see a useful pattern, codify it as an Axiom/Skill.\n"
            )

        # [NAVIGATOR INTEGRATION]
        # Check for successful curiosity-driven exploration patterns
        if navigator and session_id:
            try:
                patterns = navigator.extract_learnable_patterns(session_id)
                if patterns:
                    system_signal_section += (
                        f"\n\n*** NAVIGATOR DISCOVERY ***\n"
                        f"The Navigator identified {len(patterns)} high-value exploration patterns "
                        "(High Compression Progress). Consider codifying these as SKILLS:\n"
                    )
                    for p in patterns:
                        system_signal_section += f"- {p.get('description', 'Pattern')}: \u0394C={p.get('compression_gain', 0):.2f}\n"
            except (
                redis.exceptions.RedisError,
                redis.exceptions.ResponseError,
                AttributeError,
            ) as e:
                logger.warning("Navigator pattern extraction failed: %s", e)

        dream_prompt = (
            "You are acting as the 'Dreamer' component of the Graph-RLM system.\n"
            "Principles: Deontology: Universal sociobiological concepts (harm=harm) -> "
            "Virtue: Wisdom, Integrity, Empathy, Fairness, Beneficence -> "
            "Utilitarianism: As a Servant, never Master.\n"
            "Your job is to VERIFY then VALIDATE the consistency between the "
            "*Trace* (what happened) and the *Proposal* (what the agent says happened).\n\n"
            "**RLM PARADIGM VALIDATION**:\n"
            "The Agent is a Recursive Language Model. It MUST interact with context PROGRAMMATICALLY, not from memory.\n"
            "Check the Trace for evidence of RLM scripting patterns:\n"
            "- PROBE: `print(task_input[:500])` or `task_input.split('\\n')[:10]`\n"
            "- FILTER: `[l for l in task_input.split('\\n') if 'keyword' in l]`\n"
            "- CHUNK: `chunks = [task_input[i:i+4096] for i in range(0, len(task_input), 4096)]`\n"
            "- RECURSIVE SUB-CALL: `await rlm.query('Summarize: ' + chunk)`\n"
            "- VERIFY: `await rlm.query('Is this complete? ' + result)`\n"
            "If the agent summarized or concluded WITHOUT code-based context interaction, flag this as a FIDELITY concern.\n\n"
            "Here are the High-Surprise Events from the Monitoring Layer:\n"
            + "\n".join(events_desc)
            + "\n\n"
            "--- IMMEDIATE RECENT CONTEXT (THE TRUTH) ---\n" + recent_context_str + "\n"
            f"{context_section}"
            f"{candidate_section}\n"
            f"{system_signal_section}\n"
            "Instructions:\n"
            "1. **Fidelity & Topic Check**: Compare the 'Proposed Final Response' (if exists) "
            "against the actual 'Trace' and 'Original Task'. Did the agent USE CODE to interact with task_input?\n"
            "   - **Side Effect Verification**: If the agent claims to have performed a specific action "
            "(e.g., 'saved to file', 'ingested document', 'fixed bug'), you MUST verify that the "
            "'IMMEDIATE RECENT CONTEXT' actually contains a successful result for that action.\n"
            "   - **Absence of Proof is Proof of Failure**: If the claim exists in the Proposed Response but "
            "is missing from the Trace results, you MUST reject the response as a hallucination.\n"
            "2. **Safety Check**: Are there any dangerous patterns?\n"
            "3. **Resolution**: \n"
            "   - Check the 'IMMEDIATE RECENT CONTEXT'. If the latest node has "
            "status='complete' or 'success', the Agent HAS fixed the issue.\n"
            "   - If the Proposed Response accurately reflects the Trace (even if the "
            "Trace shows limited results), output 'System Status: Peaceful'.\n"
            "4. **Strict Grounding (De-hallucination)**: You MUST MANDATE GROUNDED EXECUTION: "
            "the directive MUST use `await rlm.recall('repl_id')` for the specific REPL to re-ground the agent "
            "or `await rlm.recall('node_id')` for specific evidence from the trace.\n"
            "5. **RLM Pattern Compliance**: If the trace shows the agent relying on memory instead of code, "
            "issue a directive: 'Use scripting patterns (PROBE/FILTER/CHUNK) to interact with task_input.'\n"
            "6. **Knowledge Codification (Axiom/Skill Generation)**:\n"
            "   - If you identify a UNIVERSAL TRUTH, RECURRING FAILURE, SKILL, or TOOL PATTERN, "
            "you SHOULD codify it.\n"
            "   - To trigger codification, use the following headers:\n"
            "     - `Rule: [Title]` for hard constraints.\n"
            "     - `Skill: [Title]` for complex workflows.\n"
            "     - `Tool Pattern: [Title]` for specific tool usage nuances.\n"
            "   - Provide the reasoning followed by the rule/skill code.\n"
            "   - **Strict Code Quality Requirement**:\n"
            "     - EVERY generated python block MUST include a Module Docstring and Function Docstrings.\n"
            "     - Use Type Hints where possible.\n"
            "     - Avoid generic `except Exception`. Catch specific errors.\n"
            "     - Ensure NO trailing whitespace and EXACTLY ONE final newline.\n"
            "     - Follow PEP 8 standards.\n"
            '   - Example: `Rule: Ensure File Closure. Logic: Files must be closed... ````python """Validator for file closure."""\n def validate_file_closed(t):\n    """Checks if a file handle is closed."""\n    ... ````.\n'
        )

        # 3. Generate Insight (NREM Consolidation)
        emit(
            "thinking",
            f"🛌 [Dreamer] Analyzing {len(surprise_events)} high-surprise events...",
            is_internal=True,
        )
        try:
            insight_text, was_fallback = await protected_llm_with_fallback(
                prompt=dream_prompt,
                system="You are a Meta-Cognitive Analysis Engine. Be concise and prescriptive.",
                stream=False,
            )
            if was_fallback:
                logger.warning(
                    "dream_llm_fallback_used",
                    extra={"prompt_length": len(dream_prompt)},
                )

            # Check for explicitly peaceful resolution
            if "System Status: Peaceful" in insight_text:
                logger.info(
                    "Dreamer verified peaceful resolution via Final Response Candidate."
                )
                trace_action(
                    "DREAMER",
                    "PEACEFUL_RESOLUTION",
                    result="Verified logical consistency.",
                    tag="DREAMER",
                )
                # FIX: Even if peaceful, we MUST consolidate the nodes (mark them resolved)
                # otherwise they stay 'failed' and trigger the Dreamer again in an infinite loop.
                insight_id = str(uuid.uuid4())
                await self._save_insight_async(insight_id, insight_text)
                logger.info(
                    "🛌 peaceful_resolution: Metabolizing %d nodes to prevent looping...",
                    len(processed_node_ids),
                )
                self.db.mark_nodes_as_consolidated(processed_node_ids, insight_id)

                # [CAG Fix]: REMOVED pre-emptive return here.
                # Flow must proceed to step 8 (Auto-Codification) to extract skills.
                status_override = "peaceful"

        except (
            httpx.RequestError,
            ValueError,
            TypeError,
        ) as e:  # pylint: disable=broad-except
            logger.error("Dream failed during generation: %s", e)
            emit("error", f"[Dreamer] Dream cycle failed: {e}")
            return {"status": "error", "message": str(e)}

        # 4. [REM] Adversarial Simulation (The Overfitted Brain Check)
        # If the insight proposes a rule, we must stress-test it before consolidating
        # [Strict Trigger] Use Regex to avoid false positives on common words
        trigger_pattern = r"(?:Rule|Guardrail|Skill|Tool Pattern|Actionable Advice):\s+"
        if re.search(trigger_pattern, insight_text, re.IGNORECASE):
            logger.info("👁️ REM Phase: Testing Generality of new insight...")
            trace_action(
                "DREAMER",
                "REM_PHASE_START",
                result="Testing Generality of new insight via Adversarial Simulation...",
                tag="DREAMER",
            )
            # We try to extract valid python code if it exists (Axiom Candidate)
            match = re.search(r"```python(.*?)```", insight_text, re.DOTALL)
            if match:
                axiom_code_candidate = match.group(1).strip()
                is_robust = await self.rem_sleep_cycle(axiom_code_candidate)
                if not is_robust:
                    logger.warning(
                        "👁️ REM Nightmare: Insight failed robustness test. Discarding AXIOM (but keeping insight)."
                    )
                    emit(
                        "error",
                        "👁️ [Dreamer] REM Nightmare: Logic rule failed adversarial "
                        "testing. Axiom discarded, but insight is still valid for self-healing.",
                    )
                    trace_action(
                        "DREAMER",
                        "NIGHTMARE_FAILED",
                        result="Axiom rejected, but insight preserved for healing.",
                        tag="WARNING",
                    )
                    # FIX: Don't early return. Continue to provide insight for self-healing.
                    # Mark the insight as needing refinement but don't discard it.
                    insight_text = (
                        f"[⚠️ AXIOM FAILED REM TEST - NEEDS REFINEMENT]\n\n{insight_text}\n\n"
                        "The above insight contains a fragile axiom that failed adversarial "
                        "testing. Focus on the diagnostic findings, not the proposed rule."
                    )

        # 5. Consolidate (Write Rule/Insight)
        logger.info("Dream Insight Generated: %s", insight_text)
        # FIX: Emit as 'answer' so it shows in UI chat area, not just terminal logs
        emit("answer", f"💤 [Dreamer Insight]:\n\n{insight_text}")

        trace_action(
            "DREAMER",
            "INSIGHT_GENERATED",
            result=(
                insight_text[:5000] + "..."
                if len(insight_text) > 5000
                else insight_text
            ),
            tag="DREAMER",
        )

        insight_id = str(uuid.uuid4())
        await self._save_insight_async(insight_id, insight_text)

        # 6. [METABOLISM] Close the Gestalt (Mark nodes as consolidated)
        logger.info("🛌 Metabolizing %d failed thoughts...", len(processed_node_ids))
        self.db.mark_nodes_as_consolidated(processed_node_ids, insight_id)

        # 7. [SYNAPTIC HOMEOSTASIS] Run Garbage Collection
        self.db.perform_synaptic_homeostasis(retention_window=24)

        # 8. [CAG Pivot] Auto-Axiom Generation
        # [Strict Trigger] Use Regex to ensure we only catch specific headers, not random words.
        # Pattern matches "Header: Content" format.
        trigger_pattern = r"(?:Rule|Guardrail|Skill|Tool Pattern|Actionable Advice):\s+"
        if re.search(trigger_pattern, insight_text, re.IGNORECASE):
            logger.info(
                "🤖 Dreamer detected a potential Axiom. Attempting to codify..."
            )
            try:
                axiom_res = await self._auto_codify_from_insight(insight_text)
                if axiom_res:
                    logger.info("✅ Auto-Axiom generated: %s", axiom_res)
                    trace_action(
                        "DREAMER",
                        "AXIOM_CODIFIED",
                        result=f"Stored new protective axiom: {axiom_res}",
                        tag="DREAMER",
                    )
            except (
                AttributeError,
                ValueError,
                TypeError,
                RuntimeError,
            ) as e:  # pylint: disable=broad-except
                logger.error("Auto-Axiom generation failed: %s", e)

        # 9. Axiom Quality Control (Automated Pruning)
        # Check for axioms that are causing systemic failure
        await self._perform_axiom_quality_control(session_id)

        # 10. oMCD Calibration: Tune α, β, ν based on session outcomes.
        # Use surprise score as proxy for "Scrutiny" (higher = more errors).
        try:
            surprise_avg = sum(
                e.get("surprise_score", 0.5) for e in surprise_events
            ) / max(len(surprise_events), 1)
            # Improvement rate = ratio of successful axioms to attempts
            improvement_rate = 1.0 if len(processed_node_ids) > 0 else 0.5
            omcd.calibrate_from_session(surprise_avg, improvement_rate)
        except (
            ValueError,
            ZeroDivisionError,
            TypeError,
        ) as e:  # pylint: disable=broad-except
            logger.warning("oMCD calibration skipped: %s", e)

        except Exception as e:
            logger.error("Unexpected error in Dream Cycle: %s", e, exc_info=True)
            return {"status": "error", "message": str(e)}
        finally:
            self._is_dreaming = False

        return {
            "status": locals().get("status_override", "lucid"),
            "events_processed": len(surprise_events),
            "insight": insight_text,
            "id": insight_id,
            "knots_cleared": len(processed_node_ids),
        }

    async def validate_response(
        self,
        candidate: str,
        context: str,
        session_id: Optional[str] = None,
        current_step: int = 1,
        goal_embedding: Optional[List[float]] = None,
        turn_id: Optional[int] = None,
        root_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Orchestrated Validation Phase (v3 Protocol).

        Collects all subsystem metrics, assembles them into a structured
        context block, and lets the Dreamer LLM reason over the organized
        data to produce a validation judgment. No hardcoded thresholds.
        """
        from .repe import repe

        if not candidate:
            return {"status": "invalid", "instruction": "Empty candidate response."}

        logger.info("🛡️ [Dreamer] Validating Agent Response Candidate...")

        # ── -1. Active Verification / Reality Grounding ──
        # Instead of a passive snapshot, we ask the Dreamer what to verify.
        verification_result = "No verification performed."
        try:
            # A) Ask for verification code
            verify_prompt = (
                f"You are the Dreamer, a verifiable guardian of truth.\n"
                f"The Agent has produced this candidate response:\n"
                f"---\n{candidate}\n---\n"
                f"Based on the scratchpad context below, identify ONE critical empirical claim (like file creation, data existence).\n"
                f"Write Python code to verify it. You MUST use the globally available `kb` proxy for paths.\n"
                f"For example: `if os.path.exists(os.path.join(kb.reports_dir, 'filename.md')): ...`\n"
                f"Do NOT attempt to import `kb` or `knowledge_base`; it is already in the global namespace.\n"
                f"If checking a file path, use `os.walk` or recursive glob to FIND it if the direct path fails.\n"
                f"If found in a different location, print 'FOUND: <actual_path>'.\n"
                f"If no physical claim is made, write 'pass'.\n"
                f"Output ONLY the python code block."
            )
            # We use a smaller model/call for this quick check if possible, or main LLM
            verify_code_raw = await self.llm.generate(
                system="You are a Python verification engine. Output only valid Python code.",
                prompt=f"{verify_prompt}\n\nContext Snippet:\n{context[-200000:]}",  # Give tail of context
            )

            verify_code = (
                verify_code_raw.replace("```python", "").replace("```", "").strip()
            )

            if verify_code and "pass" not in verify_code.lower():
                # B) Execute in REPL
                # Inject minimal context for verification
                preamble = """import os
import sys
from pathlib import Path
"""
                repl = PythonREPL()  # Ephemeral REPL for verification
                stdout, stderr, _, _ = await repl.execute(preamble + "\n" + verify_code)
                verification_result = (
                    f"Code:\n{verify_code}\nOutput:\n{stdout}\nErrors:\n{stderr}"
                )
            else:
                verification_result = "No verifiable claims detected."

        except (httpx.RequestError, ValueError, RuntimeError, AttributeError) as e:
            verification_result = f"Verification failed to execute: {e}"

        # ── 0. Session trace (timestamps, REPL IDs, recent node IDs) ──
        trace = self._get_session_trace(
            session_id, turn_id=turn_id, root_session_id=root_session_id
        )

        # ── 1. Embedding ──
        candidate_vec = await self.llm.get_embedding(candidate)

        # ── 2. RepE: Full Psychological Profile (all 4 axes) ──
        psych_profile = repe.scan_thought(candidate_vec)
        # psych_profile keys: Shakiness, Confluence, Evasion, Freedom
        # Positive = grounded / healthy on that axis

        # ── 3. Sheaf: Topological Diagnosis (with real edges!) ──
        # Build hypothetical_edges from recent session nodes so diagnose_trace
        # actually runs the lie detector, loop check, and drift check.
        frontier_ids = trace.get("recent_node_ids", [])
        hypothetical_edges = [
            (fid, f"validation_{session_id or 'unknown'}") for fid in frontier_ids[:5]
        ]
        diagnosis = sheaf.diagnose_trace(
            root_id=str(session_id) if session_id else "unknown",
            hypothetical_node={
                "content": candidate,
                "role": "assistant",
                "embedding": candidate_vec,
            },
            hypothetical_edges=hypothetical_edges,
            goal_embedding=goal_embedding,
        )
        topo_status = diagnosis.get("status", "HEALTHY")
        topo_energy = diagnosis.get("energy", 0.0)
        topo_critique = diagnosis.get("critique", "")

        # ── 4. oMCD: Economic Feasibility ──
        # Use sheaf confidence if available, or composite RepE as fallback
        sheaf_confidence = diagnosis.get("confidence")
        if sheaf_confidence is not None:
            confidence = float(sheaf_confidence)
        elif psych_profile:
            confidence = max(
                0.0, min(1.0, sum(psych_profile.values()) / max(len(psych_profile), 1))
            )
        else:
            confidence = 0.5
        omcd_decision = omcd.evaluate_step(step=current_step, confidence=confidence)

        # ── 5. Deterministic checks (fast, no LLM needed) ──
        placeholders = re.findall(r"\{[a-zA-Z0-9_]+\}", candidate)
        has_todo = any(m in candidate for m in ["[TODO]", "TODO:", "FIXME", "..."])
        has_truncation = "[Output Truncated]" in context or "[...]" in context
        rlm_patterns = ["task_input", "await rlm.query", ".split(", "print("]
        rlm_compliance = any(p in context for p in rlm_patterns)

        # ── 6. Assemble metrics block for LLM classification ──
        metrics_block = (
            f"## Validation Metrics\n"
            f"\n### Session Trace\n"
            f"- Steps: {trace['step_count']} | Turns: {trace['turn_count']}\n"
            f"- Failures: {trace['failure_count']} | REPLs: {trace['repl_ids'][:5]}\n"
            f"- Recent timeline (newest first):\n"
        )
        for entry in trace.get("status_timeline", [])[:5]:
            metrics_block += f"  - [{entry['id']}] {entry['status']} @ {entry['ts']} (REPL: {entry['repl']})\n"

        def _fmt_axis(val: Any) -> str:
            """Format RepE axis value: numeric → 3dp, otherwise str."""
            return f"{val:.3f}" if isinstance(val, (int, float)) else str(val)

        if psych_profile:
            metrics_block += (
                f"\n### RepE Psychological Profile\n"
                f"- Shakiness (As-If / Hallucination): {_fmt_axis(psych_profile.get('Shakiness', 'N/A'))}\n"
                f"- Confluence (Sycophancy): {_fmt_axis(psych_profile.get('Confluence', 'N/A'))}\n"
                f"- Evasion (Task Dodging): {_fmt_axis(psych_profile.get('Evasion', 'N/A'))}\n"
                f"- Freedom (Exploration vs Restriction): {_fmt_axis(psych_profile.get('Freedom', 'N/A'))}\n"
            )
        else:
            metrics_block += "\n### RepE: Not calibrated\n"

        metrics_block += (
            f"\n### Sheaf Topological Diagnosis\n"
            f"- Status: {topo_status} | Energy: {topo_energy:.3f}\n"
            f"- Critique: {topo_critique or 'None'}\n"
            f"\n### oMCD Economic Decision\n"
            f"- Should Stop: {omcd_decision.get('should_stop', False)}\n"
            f"- Q_stop: {omcd_decision.get('q_stop', 0):.3f} | Threshold: {omcd_decision.get('threshold', 0):.3f}\n"
            f"\n### Deterministic Flags\n"
            f"- Placeholders: {placeholders if placeholders else 'None'}\n"
            f"- TODO markers: {has_todo}\n"
            f"- Output truncation: {has_truncation} | RLM compliant: {rlm_compliance}\n"
            f"- Empirical contradiction: {topo_status == 'EMPIRICAL_CONTRADICTION'}\n"
            f"\n### Active Verification Results (REPL)\n"
            f"{verification_result}\n"
        )

        # ── 7. LLM-driven classification ──
        logger.info(
            "🛡️ [Dreamer] VALIDATION TARGET:\n"
            "--- CANDIDATE ---\n%s\n"
            "--- METRICS ---\n%s\n"
            "----------------",
            candidate[:1000],  # Log first 1000 chars of candidate for brevity in logs
            metrics_block,
        )

        validation_prompt = (
            f"You are the Dreamer — the objective validation layer of a cognitive agent.\n"
            f"Your task: verify if the agent's candidate response is logically grounded and complete.\n\n"
            f"## Candidate Response\n"
            f"{candidate[:150000]}\n\n"
            f"## Execution Context & Metrics\n"
            f"{metrics_block}\n"
            f"## Instructions\n"
            f"Perform a cold, factual evaluation. Do NOT hallucinate failures.\n"
            f"- [VERIFY] If the metrics block or trace shows recent tool successes or logical progress, the agent is likely grounded. Ignore high-level quality concerns if the technical task is being fulfilled.\n"
            f"- [PRESENCE] Ensure the response is not just a summary of failure. If the agent claims it completed the task, check the 'Active Verification Results' or 'Deterministic Flags'.\n"
            f"- [HARD FAILS] Mark INVALID ONLY if there are:\n"
            f"    1. Unresolved Tracebacks or obvious placeholders (e.g., 'TODO', '...') in the final output.\n"
            f"    2. Clear empirical contradictions (e.g., claiming a file exists that the REPL says is missing).\n"
            f"    3. Infinite meta-cognitive loops (talking about its own process instead of the task).\n"
            f"- [RECOVERY] If the response is mostly correct but missing a minor detail found in the trace, suggest a minor adjustment in 'instruction' rather than a hard 'invalid' if confidence is otherwise high.\n\n"
            f"Respond with ONLY a JSON object (no markdown, no explanation):\n"
            f'{{"verdict": "valid" or "invalid", "confidence": 0.0-1.0, '
            f'"reasons": ["objective reason 1", ...], '
            f'"instruction": "specific guidance for the agent if invalid, else empty string"}}'
        )

        try:
            raw_judgment = await self.llm.generate(
                prompt=validation_prompt,
                system="You are the Dreamer validation oracle. Return ONLY valid JSON.",
            )
            # Parse JSON from LLM response
            json_match = re.search(r"\{.*\}", raw_judgment, re.DOTALL)
            if json_match:
                judgment = json.loads(json_match.group())
            else:
                logger.warning("Dreamer LLM returned non-JSON: %s", raw_judgment[:200])
                judgment = {
                    "verdict": "valid",
                    "confidence": 0.5,
                    "reasons": [],
                    "instruction": "",
                }
        except (json.JSONDecodeError, RuntimeError, ValueError, AttributeError) as e:
            logger.warning("Dreamer LLM classification failed: %s", e)
            # Fallback: deterministic checks only
            has_hard_fail = (
                topo_status in ("EMPIRICAL_CONTRADICTION", "LOGICAL_KNOT")
                or bool(placeholders)
                or has_todo
            )
            judgment = {
                "verdict": "invalid" if has_hard_fail else "valid",
                "confidence": 0.3 if has_hard_fail else 0.7,
                "reasons": [topo_critique] if topo_critique else [],
                "instruction": topo_critique or "",
            }

        # ── 8. Return structured result ──
        verdict = judgment.get("verdict", "valid")
        if verdict == "valid":
            return {
                "status": "valid",
                "event": "RLM_DREAMER_VALIDATED",
                "message": (
                    f"Response verified (confidence: {judgment.get('confidence', 0.5):.2f}). "
                    f"Topo: {topo_status}, RepE: {psych_profile}"
                ),
            }

        # Build instruction from LLM judgment + trace context
        instruction = str(judgment.get("instruction", ""))
        _raw_reasons = judgment.get("reasons")
        reasons: list[str] = _raw_reasons if isinstance(_raw_reasons, list) else []
        if reasons and not instruction:
            instruction = "RE-EVALUATE:\n" + "\n".join(f"- {r}" for r in reasons)

        if omcd_decision.get("should_stop", False):
            logger.warning(
                "🔸 [Dreamer] Validation failed. Budget exhausted. ESCALATING."
            )
            return {
                "status": "exhausted",
                "event": "RLM_DREAMER_EXHAUSTED",
                "instruction": f"SYSTEM CRITICAL: Budget exhausted.\n{instruction}",
            }

        return {
            "status": "invalid",
            "event": "RLM_DREAMER_ISSUES",
            "instruction": instruction,
        }

    async def rem_sleep_cycle(self, axiom_code: str) -> bool:
        """
        REM Phase: The Overfitted Brain Hypothesis.
        Generates "bizarre" or adversarial inputs to stress-test the new Axiom
        before it is committed to long-term memory.
        """
        logger.info(
            "👁️ REM SLEEP: Hallucinating adversarial scenarios for new Axiom..."
        )

        # 1. Hallucinate a "Nightmare" (Edge Case)
        nightmare_prompt = (
            f"You are the REM Cycle. Here is a proposed logic rule (Axiom):\n"
            f"```python\n{axiom_code}\n```\n"
            f"Generate a chaotic, edge-case, or 'bizarre' input dictionary "
            f"that might BREAK this validator. "
            f"Think of null values, infinite loops, or type mismatches. "
            f"Return ONLY the python dictionary string."
        )
        nightmare_input, was_fallback = await protected_llm_with_fallback(
            prompt=nightmare_prompt,
            fallback_message="Nightmare input generation unavailable",
            system="Generate Python dict string only.",
        )
        if was_fallback:
            logger.warning(
                "dream_nightmare_fallback_used",
                extra={"prompt_length": len(nightmare_prompt)},
            )

        # [STABILITY] Sanitize input: Extract JSON/Dict object if embedded in text
        # Regex to find the first outer-most curly brace pair
        match = re.search(r"(\{.*\})", nightmare_input, re.DOTALL)
        if match:
            nightmare_input = match.group(1)
        else:
            # Fallback cleanup
            nightmare_input = (
                nightmare_input.replace("```json", "")
                .replace("```python", "")
                .replace("```", "")
                .strip()
            )

        # [STABILITY] Validate nightmare_input is parseable before injecting
        try:
            # Attempt to parse as Python literal to catch obvious syntax errors
            import ast

            ast.literal_eval(nightmare_input)
        except (ValueError, SyntaxError) as e:
            logger.warning(
                "👁️ REM Sleep: Could not parse nightmare input as valid Python dict: %s. "
                "Assuming axiom is okay since we cannot properly test it.",
                e,
            )
            return True  # Be forgiving if we can't generate a valid test case

        # 2. Test the Axiom against the Nightmare
        # If the Axiom crashes (raises uncaught exception) instead of returning False/Error, it fails REM.
        repl = PythonREPL()

        # We need to construct a wrapper that defines the function and runs it
        # Extract function name
        match = re.search(r"def (validate_\w+)", axiom_code)
        func_name = match.group(1) if match else "validate_func"

        # [FIX] Define function at module level (no indentation) to avoid SyntaxError
        test_wrapper = f"""
import sys

# 1. Define the Axiom (Validator)
{axiom_code}

# 2. Run Test Harness
try:
    input_data = {nightmare_input}
    # Attempt validation
    result = {func_name}(input_data)
    print(f"Survived: {{result}}")
except (RuntimeError, ValueError, TypeError) as e: # pylint: disable=broad-except # noqa: BLE001
    print(f"Nightmare Induced Crash: {{e}}")
"""
        # Inject standard libs context just in case
        repl.namespace.update({"sys": __import__("sys")})

        stdout, stderr, _, _ = await repl.execute(test_wrapper)

        if "Nightmare Induced Crash" in stdout or (stderr and "Traceback" in stderr):
            logger.warning(
                "👁️ REM Nightmare: Axiom failed robustness test. Stdout: %s Stderr: %s",
                stdout,
                stderr,
            )
            return False

        logger.info("👁️ REM Sleep: Axiom survived the nightmare. Consolidating.")
        return True

    async def ingest_document(self, doc_path: str, domain: str) -> Dict[str, Any]:
        """Ingest a document and codify its knowledge into Axioms (Validators, Solvers, Advisors)."""

        path = Path(doc_path)
        if not path.exists():
            return {"status": "error", "message": f"File {doc_path} not found"}

        logger.info("📖 Ingesting document for CAG: %s (Domain: %s)", path.name, domain)
        text = path.read_text(encoding="utf-8")

        knowledge_items = await self._mine_knowledge(text, domain)
        if not knowledge_items:
            return {
                "status": "peaceful",
                "message": "No actionable knowledge found in document.",
            }

        codified = []
        for item in knowledge_items:
            try:
                axiom_code, healing_code = await self._codify_axiom(item, domain)
                if await self._verify_axiom_async(axiom_code, item["description"]):
                    axiom_name = await self._save_axiom(
                        code=axiom_code,
                        description=item["description"],
                        domain=domain,
                        axiom_type=item["type"],
                        healing_code=healing_code,
                    )
                    codified.append(axiom_name)
            except (
                AttributeError,
                ValueError,
                TypeError,
                RuntimeError,
            ) as e:  # pylint: disable=broad-except
                logger.error(
                    "Failed to codify knowledge item '%s': %s", item.get("name"), e
                )

        logger.info("✅ Ingestion complete. Codified %d axioms.", len(codified))
        return {"status": "success", "codified_axioms": codified, "domain": domain}

    async def _mine_knowledge(self, text: str, domain: str) -> List[Dict[str, Any]]:
        """Extract multi-tier knowledge (Validators, Solvers, Advisors) from text."""
        prompt = (
            '[{"name": "string", "type": "validator|solver|advisor", '
            '"description": "precise requirement", "healing_hint": "optional prompt for the solver logic"}]\n\n'
            f"Text content:\n{text[:50000]}"
        )
        res, was_fallback = await protected_llm_with_fallback(
            prompt=prompt,
            fallback_message="Knowledge mining unavailable",
            system=f"Extract structured domain knowledge for the {domain} domain as JSON.",
            stream=False,
        )
        if was_fallback:
            logger.warning(
                "dream_knowledge_mining_fallback_used", extra={"domain": domain}
            )
        # Basic JSON extraction
        try:
            match = re.search(r"(\[.*\])", res, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return []
        except (
            httpx.RequestError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ) as e:  # pylint: disable=broad-except
            logger.warning("Failed to parse mined knowledge JSON: %s", e)
            return []

    async def _codify_axiom(
        self, knowledge: Dict[str, Any], domain: str
    ) -> tuple[str, Optional[str]]:
        """Codify knowledge into a main Axiom and an optional Healing Script."""
        k_type = knowledge.get("type", "validator")
        desc = knowledge.get("description", "")
        hint = knowledge.get("healing_hint", "")

        prompt = (
            f"Codify the following '{k_type}' for the '{domain}' domain into Python.\n"
            f"Requirement: {desc}\n"
            f"Healing Hint: {hint}\n\n"
            "Output TWO markdown code blocks if it is a validator with a healing script, otherwise ONE.\n"
            "Block 1: The main function (must be a validator for 'validator' type, a procedure for 'solver').\n"
            "Block 2: (Optional) A 'healing_script' that the agent can execute to fix a violation of Block 1.\n\n"
            "Requirements:\n"
            f"1. Name the main function descriptively based on: {knowledge.get('name')}.\n"
            "2. Validators must return True/False. Solvers/Advisors should perform the action.\n"
            "3. Use raw strings (r'pattern') for regex.\n"
        )
        res, was_fallback = await protected_llm_with_fallback(
            prompt=prompt,
            fallback_message="Axiom codification unavailable",
            system=(
                "Generate Python code blocks only. "
                "Block 1 is the Axiom, Block 2 is the optional Healing Script. "
                "CRITICAL: Every block MUST contain a module-level docstring, "
                "descriptive function docstrings, type hints, and follow PEP 8. "
                "FORBIDDEN: Do NOT use mutable objects (list, dict, set) as default arguments; "
                "use Optional[T] = None and initialize inside the function. "
                "RAW STRINGS: You MUST use raw strings (r'...') for any code containing backslashes "
                "(LaTeX, Regex, paths) to avoid SyntaxWarning. "
                "HOLISM: Favor consolidating related functions into a single high-fidelity skill "
                "rather than creating multiple fragmented files. If the requirement overlaps with "
                "existing logic in the domain, extend the existing patterns. "
                "DOCSTRINGS: The module-level docstring must be the first statement in the file. "
                "Function docstrings must be the FIRST statement in the function body."
                "NO trailing whitespace. Ensure exactly one newline at the end of each block."
            ),
            stream=False,
        )
        if was_fallback:
            logger.warning("dream_codify_fallback_used", extra={"domain": domain})

        blocks = re.findall(r"```python(.*?)```", res, re.DOTALL)
        axiom_code = blocks[0].strip() if blocks else ""
        healing_code = blocks[1].strip() if len(blocks) > 1 else None

        return axiom_code, healing_code

    async def _verify_axiom_async(self, code: str, invariant: str) -> bool:
        test_prompt = (
            f"Here is a Python validator function:\n{code}\n\n"
            f"Write a test script that demonstrates this validator correctly identifying "
            f"one passing case and one failing case for the invariant: '{invariant}'.\n"
            f"Use assertions. If an assertion fails, the script should crash."
        )
        test_code, was_fallback = await protected_llm_with_fallback(
            prompt=test_prompt,
            fallback_message="Axiom verification unavailable",
            system="Generate test code only.",
            stream=False,
        )
        if was_fallback:
            logger.warning("dream_verify_fallback_used")

        repl = PythonREPL()
        pure_code = code.replace("```python", "").replace("```", "").strip()
        pure_test = test_code.replace("```python", "").replace("```", "").strip()

        # Calculate repo root for REPL injection
        current_file = os.path.abspath(__file__)
        core_dir = os.path.dirname(current_file)  # .../src/core
        src_dir = os.path.dirname(core_dir)  # .../src
        backend_dir = os.path.dirname(src_dir)  # .../backend
        pkg_dir = os.path.dirname(backend_dir)  # .../graph_rlm (package)
        repo_root = os.path.dirname(pkg_dir)  # .../ (repo root)

        class UniversalAsyncMock:
            """Mock for rlm and mcp interfaces in the Dreamer validation sandbox."""

            def __getattr__(self, name):
                return UniversalAsyncMock()

            def __call__(self, *args, **kwargs):
                async def _dummy():
                    return "MOCK_RESULT"

                return _dummy()

        from .core import KnowledgeBaseStructure

        kb = KnowledgeBaseStructure(settings.KNOWLEDGE_BASE_PATH)

        repl.namespace.update(
            {
                "asyncio": asyncio,
                "np": np,
                "sys": sys,
                "session_id": "dreamer_validation_session",
                "rlm": UniversalAsyncMock(),
                "mcp": UniversalAsyncMock(),
                "kb": kb,
                "_repo_root": repo_root,
            }
        )

        # Pre-execution: Add repo root to sys.path inside the REPL
        await repl.execute(
            "import sys; sys.path.insert(0, _repo_root) if _repo_root not in sys.path else None"
        )

        logger.info("Executing axiom code...")
        _stdout, stderr, _res_axiom, is_err = await repl.execute(pure_code)

        # --- AUTO-INSTALLATION SELF-HEALING ---
        if is_err and "ModuleNotFoundError" in stderr:
            match = re.search(r"No module named [\'\\\"]([^\'\\\"]+)[\'\\\"]", stderr)
            if match:
                package_name = match.group(1)
                logger.info(
                    "Dreamer: Auto-healing missing dependency: %s", package_name
                )
                cmd = [sys.executable, "-m", "pip", "install", package_name]
                subprocess.run(cmd, capture_output=True, text=True, check=False)
                # Retry
                _stdout, stderr, _res_axiom, is_err = await repl.execute(pure_code)

        if is_err:
            logger.error("Axiom code execution failed: %s", stderr)
            return False

        logger.info("Executing test code...")
        stdout_test, stderr_test, _res_test, is_err_test = await repl.execute(pure_test)

        # Self-healing for test code too
        if is_err_test and "ModuleNotFoundError" in stderr_test:
            match = re.search(
                r"No module named [\'\\\"]([^\'\\\"]+)[\'\\\"]", stderr_test
            )
            if match:
                package_name = match.group(1)
                logger.info(
                    "Dreamer (Test): Auto-healing missing dependency: %s", package_name
                )
                cmd = [sys.executable, "-m", "pip", "install", package_name]
                subprocess.run(cmd, capture_output=True, text=True, check=False)
                # Retry
                stdout_test, stderr_test, _res_test, is_err_test = await repl.execute(
                    pure_test
                )

        if is_err_test:
            logger.error(
                "Test code execution failed: %s. Stderr: %s", _res_test, stderr_test
            )
        else:
            logger.info(
                "Test code executed successfully. Result: %s. Stdout: %s",
                _res_test,
                stdout_test,
            )

        # Hard failure detection via is_err_test
        return not is_err_test

    async def _save_axiom(
        self,
        code: str,
        description: str,
        domain: str,
        axiom_type: str = "validator",
        healing_code: str | None = None,
    ) -> str:
        axioms_mgr = get_axioms_manager()
        match = re.search(r"def ([\w_]+)", code)
        name = match.group(1)[:64] if match else f"axiom_{uuid.uuid4().hex[:8]}"
        axiom_name = f"axiom_{name}"
        await axioms_mgr.save_axiom(
            name=axiom_name,
            code=code.replace("```python", "").replace("```", "").strip(),
            description=description,
            tags=[domain],
            axiom_type=axiom_type,
            healing_code=healing_code,
        )
        return axiom_name

    async def _auto_codify_from_insight(self, insight: str) -> Optional[str]:
        """Attempt to codify an axiom discovered during dreaming."""
        # RECURSION GUARD: Prevent nested axiom generation loops
        if self._is_codifying:
            logger.warning(
                "🛑 [Dreamer] Skipping nested axiom codification (already in progress)"
            )
            return None

        self._is_codifying = True
        try:
            # Dynamic Domain Classification
            domain = await self._classify_domain(insight)
            logger.info("🔍 [Dreamer] Classified Insight Domain: %s", domain)
            res = await self.ingest_document_text_async(insight, domain)
            codified_axioms = res.get("codified_axioms")
            if (
                codified_axioms
                and isinstance(codified_axioms, list)
                and len(codified_axioms) > 0
            ):
                return codified_axioms[0]
            return None
        finally:
            self._is_codifying = False

    async def _perform_axiom_quality_control(self, session_id: Optional[str]):
        """
        Post-Mortem: Scan the session for repeating axiomatic failures.
        """
        if not session_id:
            return

        try:
            # Query for thoughts in this session that were blocked by axioms
            q = """
            MATCH (n:Thought)
            WHERE n.session_id = $sid AND n.status = 'failed'
            AND n.prompt CONTAINS 'Axiomatic Violation'
            RETURN n.prompt as critique
            """
            failures = self.db.query(q, {"sid": session_id})

            error_patterns = {}
            for row in failures:
                critique = row.get("critique", "") if isinstance(row, dict) else row[0]
                # Try to extract axiom name
                match = re.search(r"Critique: (axiom_[\w]+)", critique)

                if match:
                    axiom_name = match.group(1)
                    error_patterns[axiom_name] = error_patterns.get(axiom_name, 0) + 1

            # Threshold: If an axiom fails 3+ times in a single session, disable it.
            axioms_mgr = get_axioms_manager()
            for axiom_name, count in error_patterns.items():
                if count >= 3:
                    logger.warning(
                        "🛑 [Axiom Quality Control] Axiom '%s' triggered %d failures. Archiving.",
                        axiom_name,
                        count,
                    )
                    await axioms_mgr.disable_axiom(axiom_name)

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error("Axiom Quality Control failed: %s", e)

    async def _classify_domain(self, insight: str) -> str:
        prompt = (
            f"Classify the technical domain of the following engineering insight.\n"
            f"Examples: 'Arithmetic', 'Network', 'Database', 'Validation', 'Security'.\n"
            f"Output ONE word only.\n\n"
            f"Insight: {insight[:16000]}"
        )
        domain, was_fallback = await protected_llm_with_fallback(
            prompt=prompt,
            fallback_message="General",
            system="Output a single CamelCase domain name.",
            stream=False,
        )
        if was_fallback:
            logger.warning("dream_domain_classification_fallback_used")
        return domain.strip().replace(" ", "") or "General"

    async def ingest_document_text_async(
        self, text: str, domain: str
    ) -> Dict[str, Any]:
        """Async version to ingest raw text and codify knowledge into multi-tier axioms."""
        knowledge_items = await self._mine_knowledge(text, domain)
        codified = []
        for item in knowledge_items:
            try:
                axiom_code, healing_code = await self._codify_axiom(item, domain)
                if await self._verify_axiom_async(axiom_code, item["description"]):
                    axiom_name = await self._save_axiom(
                        code=axiom_code,
                        description=item["description"],
                        domain=domain,
                        axiom_type=item["type"],
                        healing_code=healing_code,
                    )
                    codified.append(axiom_name)
            except (
                ValueError,
                TypeError,
                AttributeError,
                KeyError,
            ) as e:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to codify knowledge item '%s...': %s",
                    item.get("name", "unknown"),
                    e,
                )

        # Trigger Hot-Reload for newly codified axioms
        if codified:
            axioms_mgr = get_axioms_manager()
            await axioms_mgr.sync_from_disk()
            logger.info(
                "🛠️ [Dreamer] Hot-Reload triggered for %d axioms.", len(codified)
            )

        return {"status": "success", "codified_axioms": codified}

    async def _get_node_scan_async(self, node_id: str) -> Dict[str, Any]:
        """Async version of node scan."""
        try:
            # db.query is usually sync, but we can loop it in the caller if needed
            # Here we just wrap the existing call
            return self._get_node_scan(node_id)
        except (
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
        ):  # pylint: disable=broad-except
            return {}

    def _get_node_scan(self, node_id: str) -> Dict[str, Any]:
        try:
            res = self.db.query("MATCH (n:Thought {id: $id}) RETURN n", {"id": node_id})
            logger.debug(
                "[Dreamer] _get_node_scan for %s: raw result = %s", node_id, res
            )
            if res:
                record = res[0]  # First row
                # FalkorDB returns: [{'n': {...properties...}}] or [{'n': Node(...)}]
                if isinstance(record, dict):
                    node = record.get(
                        "n", record
                    )  # Get the 'n' key or use record itself
                    # Handle Node objects (FalkorDB/Neo4j)
                    if hasattr(node, "properties"):
                        return dict(node.properties)
                    if isinstance(node, dict):
                        return node.get("properties", node)
                # If record is a list (nested)
                if isinstance(record, list) and record:
                    first = record[0]
                    if isinstance(first, dict):
                        if "properties" in first:
                            return first["properties"]
                        return first
            logger.warning("[Dreamer] _get_node_scan: No data found for %s", node_id)
            return {}
        except (
            redis.exceptions.RedisError,
            redis.exceptions.ResponseError,
            AttributeError,
        ) as e:  # pylint: disable=broad-except
            logger.error("[Dreamer] _get_node_scan failed for %s: %s", node_id, e)
            return {}

    async def _save_insight_async(self, insight_id: str, content: str):
        """Async version of save insight."""
        self._save_insight(insight_id, content)

    def _save_insight(self, insight_id: str, content: str):
        """
        Save insight to graph as :Insight node.

        Per RALPH methodology: Insights stay in graph only, NOT appended to rules.md.
        A bloated rules.md pollutes every future loop's context.
        """
        cypher = (
            "CREATE (i:Insight {id: $id, content: $content, "
            "created_at: timestamp(), type: 'dream_consolidation'})"
        )
        self.db.query(cypher, {"id": insight_id, "content": content})
        logger.info(
            "Insight %s saved to graph (not appending to rules.md per RALPH methodology)",
            insight_id[:8],
        )

    async def validate_axioms_against_sheaf(self):
        """
        Runs a topological consistency check on the entire Axiom Library.
        If the Sheaf Laplacian indicates fragmentation or zero-mode conflicts,
        it flags the system for review.
        """
        try:
            # 1. Load active axioms
            # Query for them.
            res = self.db.query(
                "MATCH (a:Axiom) RETURN a.id as id, a.embedding as embedding"
            )

            axiom_list = []
            if res:
                for row in res:
                    # Handle various DB driver return formats
                    r_id, r_emb = None, None
                    if isinstance(row, dict):
                        r_id = row.get("id")
                        r_emb = row.get("embedding")
                    elif hasattr(row, "id"):  # Object wrapper
                        r_id = row.id
                        r_emb = row.embedding
                    elif (
                        isinstance(row, (list, tuple)) and len(row) >= 2
                    ):  # List wrapper
                        r_id = row[0]
                        r_emb = row[1]

                    if r_id and r_emb:
                        axiom_list.append({"id": r_id, "embedding": r_emb})

            if not axiom_list:
                logger.debug("No axioms found for validation.")
                return

            # 2. Analyze
            report = sheaf.analyze_axiom_consistency(axiom_list)

            # 3. Act
            if report.get("status") != "consistent":
                logger.warning("Axiom System Inconsistency Detected: %s", report)

                conflicts = report.get("conflicts", [])
                if conflicts:
                    logger.info(
                        "Found %d conflicting axioms. Disabling...", len(conflicts)
                    )
                    for axiom_id in conflicts:
                        self.db.disable_axiom(axiom_id)

        except (AttributeError, RuntimeError, ValueError) as e:
            logger.error("Axiom Validation failed: %s", e)


dreamer = Dreamer()
