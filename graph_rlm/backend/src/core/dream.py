"""
The Dreamer: Consolidation and Wisdom layer for Graph-RLM.
Analyzes surprise events, logical knots, and codifies axioms during 'sleep' cycles.
"""

import re
import uuid
from typing import Any, Dict, List, Optional

from .core import PythonREPL
from .db import GraphClient, db
from .llm import llm
from .logger import get_logger
from .sheaf import sheaf

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

    async def analyze_holonomy(
        self, loop_nodes: List[Dict[str, Any]], current_thought: str
    ) -> str:
        """
        [Lucid Dream] Immediate synchronous analysis of a detected logical knot.
        Explains WHY the agent is looping and prescribes a specific exit strategy.
        """
        logger.info("⚡ [Dreamer] Triggering Lucid Dream for Holonomy Analysis...")

        # Format history trace
        trace_str = ""
        for i, node in enumerate(reversed(loop_nodes)):  # Most recent first usually
            # Handle FalkorDB/Neo4j node structures which might be dicts or objects
            props = node
            if hasattr(node, "properties"):
                props = node.properties
            elif "n" in node:
                props = node["n"]

            # Normalize dict access
            if hasattr(props, "properties"):
                props = props.properties

            content = props.get("content", str(props))
            trace_str += f"Step -{i}: {content[:300]}...\n"

        # Find the "Parent of Divergence" (The knot's first node)
        divergence_node_id = loop_nodes[-1].get("id") if loop_nodes else "unknown"

        prompt = (
            "You are the Meta-Cognitive Supervisor (The Dreamer).\n"
            "The Agent is stuck in a LOGICAL KNOT (Infinite Loop). It keeps repeating the same semantic thought.\n\n"
            f"--- LOOP TRACE ---\n{trace_str}\n"
            f"--- CURRENT THOUGHT ---\n{current_thought[:500]}\n\n"
            f"--- DIVERGENCE POINT ---\nNode ID: {divergence_node_id}\n\n"
            "Task:\n"
            "1. Identify the specific variable, assumption, or action that is causing the repetition.\n"
            "2. Provide a CRITICAL, 1-SENTENCE directive to break the loop.\n"
            "3. You MUST mandate the use of `await rlm.recall(repl_id)` if the agent is hallucinating state.\n"
            "4. Start with 'BREAK LOOP:'."
        )

        try:
            analysis = await self.llm.generate(
                prompt=prompt,
                system="You are an emergency loop-breaker intervention system.",
                stream=False,
            )
            logger.info("⚡ [Dreamer] Holonomy Analysis: %s", analysis)
            return analysis
        except Exception as e:
            logger.error("Dreamer analysis failed: %s", e)
            return "BREAK LOOP: You are repeating yourself. Stop whatever you are doing and try a completely different approach."

    async def dream_cycle(
        self,
        emit_callback=None,
        session_id: Optional[str] = None,
        final_response_candidate: Optional[str] = None,
        context: Optional[str] = None,
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
            final_response_candidate: The Agent's proposed final answer. Dreamer checks if this resolves the failures.
            context: The full Agent Scratchpad (history, REPL IDs, recent topology) for cross-verification.
        """

        def emit(event_type, content):
            if emit_callback:
                emit_callback(event_type, content)

        logger.info("🛌 Initiating Dream Cycle (Sleep Phase)...")
        emit("thinking", "🛌 [Dreamer] Initiating Dream Cycle...")

        # 1. Gather Surprise (High Energy Edges) - scoped to current session if provided
        surprise_events = sheaf.compute_sheaf_surprise_score(
            limit=10, session_id=session_id
        )

        if not surprise_events:
            logger.info("No high-surprise events found. Sleep was peaceful.")
            return {"status": "peaceful", "insights": []}

        logger.info("Found %d high-surprise events.", len(surprise_events))

        processed_node_ids = [e["target"] for e in surprise_events]

        # [NEW] 1. Gather Recent Frontier (The Truth)
        # We need to see the *latest* events to check for success
        recent_context_str = "No recent context"
        if session_id:
            recent_events = self.db.query(
                """
                MATCH (n:Thought)
                WHERE n.session_id = $sid
                RETURN n.id as id, n.prompt as prompt, n.status as status, n.result as result
                ORDER BY n.created_at DESC LIMIT 5
                """,
                {"sid": session_id},
            )

            if recent_events:
                # Normalize result formatting
                recent_lines = []
                for r in recent_events:
                    rid = r.get("id", "???")
                    status = r.get("status", "unknown")
                    prompt = str(r.get("prompt") or "")[:50]
                    res = str(r.get("result") or "")[:100]
                    recent_lines.append(
                        f"- [Node {rid}] Status: {status} | Action: {prompt}... | Result: {res}..."
                    )
                recent_context_str = "\n".join(recent_lines)

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
                f"  Parent Thought: {src_node.get('prompt', 'Unknown')[:100]}...\n"
                f"  Child Action: {tgt_node.get('prompt', 'Unknown')[:100]}...\n"
                f"  Result: {tgt_node.get('result', 'Unknown')[:200]}..."
            )

        candidate_section = ""
        if final_response_candidate:
            candidate_section = (
                f"\n\n--- AGENT PROPOSED FINAL RESPONSE ---\n"
                f"{final_response_candidate[:2000]}...\n"
                f"---------------------------------------\n"
            )

        context_section = ""
        if context:
            context_section = (
                f"\n\n--- AGENT SCRATCHPAD & HISTORY (CONTEXT) ---\n"
                f"{context}\n"
                f"--------------------------------------------\n"
            )

        dream_prompt = (
            "You are acting as the 'Dreamer' component of the Graph-RLM system.\n"
            "The Graph-RLM system is a multi-agent system that uses a graph to store and retrieve information.\n"
            "Principles: Deontology: Universal sociobiological concepts (harm=harm) -> Virtue: Wisdom, Integrity, Empathy, Fairness, Beneficence -> Utilitarianism: As a Servant, never Master.\n"
            "Your job is to VERIFY then VALIDATE the consistency between the *Trace* (what happened) and the *Proposal* (what the agent says happened).\n\n"
            "Here are the High-Surprise Events from the Monitoring Layer:\n"
            + "\n".join(events_desc)
            + "\n\n"
            "--- IMMEDIATE RECENT CONTEXT (THE TRUTH) ---\n" + recent_context_str + "\n"
            f"{context_section}"
            f"{candidate_section}\n"
            "Instructions:\n"
            "1. **Trace Fidelity Check**: Compare the 'Proposed Final Response' (if exists) against the actual 'Trace/Scratchpad'.\n"
            "   - Did the Agent report tools/data that are NOT in the trace? (Hallucination -> REJECT)\n"
            "   - Did the Agent report the CORRECT tools/data found in the trace? (Fidelity -> ACCEPT)\n"
            "2. **Safety Check**: Are there any dangerous patterns?\n"
            "3. **Resolution**: \n"
            "   - Check the 'IMMEDIATE RECENT CONTEXT'. If the latest node has status='complete' or 'success', the Agent HAS fixed the issue.\n"
            "   - Do NOT reject based solely on past 'High-Surprise Events' if the 'Recent Context' shows resolution.\n"
            "   - If the Proposed Response accurately reflects the Trace (even if the Trace shows limited results), output 'System Status: Peaceful'.\n"
            "   - Only reject if there is a contradiction (e.g. Agent says 'I used Google' but Trace shows 'Error').\n"
            "   - Do NOT invent failures. If the agent listed the tools correctly, do not claim it didn't.\n"
            "   - MANDATE GROUNDED EXECUTION: If you detect a contradiction, the directive MUST use `await rlm.recall(repl_id)` for the specific REPL to re-ground the agent.\n"
        )

        # 3. Generate Insight (NREM Consolidation)
        emit(
            "thinking",
            f"🛌 [Dreamer] Analyzing {len(surprise_events)} high-surprise events...",
        )
        try:
            insight_text = await self.llm.generate(
                prompt=dream_prompt,
                system="You are a Meta-Cognitive Analysis Engine. Be concise and prescriptive.",
                stream=False,
            )

            # Check for explicitly peaceful resolution
            if "System Status: Peaceful" in insight_text:
                logger.info(
                    "Dreamer verified peaceful resolution via Final Response Candidate."
                )
                return {"status": "peaceful", "insights": [], "message": insight_text}

        except Exception as e:
            logger.error("Dream failed during generation: %s", e)
            emit("error", f"[Dreamer] Dream cycle failed: {e}")
            return {"status": "error", "message": str(e)}

        # 4. [REM] Adversarial Simulation (The Overfitted Brain Check)
        # If the insight proposes a rule, we must stress-test it before consolidating
        triggers = ["Rule:", "Guardrail:", "Guardrail Rule", "Actionable Advice"]
        if any(t in insight_text for t in triggers):
            logger.info("👁️ REM Phase: Testing Generality of new insight...")
            # We try to extract valid python code if it exists (Axiom Candidate)
            match = re.search(r"```python(.*?)```", insight_text, re.DOTALL)
            if match:
                axiom_code_candidate = match.group(1).strip()
                is_robust = await self.rem_sleep_cycle(axiom_code_candidate)
                if not is_robust:
                    logger.warning(
                        "👁️ REM Nightmare: Insight failed robustness test. Discarding."
                    )
                    emit(
                        "error",
                        "👁️ [Dreamer] REM Nightmare: Logic rule failed adversarial testing. Discarded.",
                    )
                    return {
                        "status": "nightmare_failed",
                        "message": "Axiom was too fragile.",
                    }

        # 5. Consolidate (Write Rule/Insight)
        logger.info("Dream Insight Generated: %s", insight_text)
        emit("thinking", f"💤 [Dreamer Insight]: {insight_text}")

        insight_id = str(uuid.uuid4())
        await self._save_insight_async(insight_id, insight_text)

        # 6. [METABOLISM] Close the Gestalt (Mark nodes as consolidated)
        logger.info("🛌 Metabolizing %d failed thoughts...", len(processed_node_ids))
        self.db.mark_nodes_as_consolidated(processed_node_ids, insight_id)

        # 7. [SYNAPTIC HOMEOSTASIS] Run Garbage Collection
        self.db.perform_synaptic_homeostasis(retention_window=24)

        # 8. [CAG Pivot] Auto-Axiom Generation
        if any(t in insight_text for t in triggers):
            logger.info(
                "🤖 Dreamer detected a potential Axiom. Attempting to codify..."
            )
            try:
                axiom_res = await self._auto_codify_from_insight(insight_text)
                if axiom_res:
                    logger.info("✅ Auto-Axiom generated: %s", axiom_res)
            except Exception as e:
                logger.error("Auto-Axiom generation failed: %s", e)

        return {
            "status": "lucid",
            "events_processed": len(surprise_events),
            "insight": insight_text,
            "id": insight_id,
            "knots_cleared": len(processed_node_ids),
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
            f"Generate a chaotic, edge-case, or 'bizarre' input dictionary that might BREAK this validator. "
            f"Think of null values, infinite loops, or type mismatches. "
            f"Return ONLY the python dictionary string."
        )
        nightmare_input = await self.llm.generate(
            nightmare_prompt, system="Generate Python dict string only."
        )

        # 2. Test the Axiom against the Nightmare
        # If the Axiom crashes (raises uncaught exception) instead of returning False/Error, it fails REM.
        repl = PythonREPL()

        # We need to construct a wrapper that defines the function and runs it
        # Extract function name
        match = re.search(r"def (validate_\w+)", axiom_code)
        func_name = match.group(1) if match else "validate_func"

        test_wrapper = f"""
import sys
try:
{axiom_code}
    input_data = {nightmare_input}
    # Attempt validation
    result = {func_name}(input_data)
    print(f"Survived: {{result}}")
except Exception as e:
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
        from pathlib import Path

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
            except Exception as e:
                logger.error(
                    "Failed to codify knowledge item '%s': %s", item.get("name"), e
                )

        logger.info("✅ Ingestion complete. Codified %d axioms.", len(codified))
        return {"status": "success", "codified_axioms": codified, "domain": domain}

    async def _mine_knowledge(self, text: str, domain: str) -> List[Dict[str, Any]]:
        """Extract multi-tier knowledge (Validators, Solvers, Advisors) from text."""
        prompt = (
            f"You are the 'Knowledge Miner' for a Domain-Specific Axiom system (Domain: '{domain}').\n"
            "Analyze the text and extract three types of knowledge:\n"
            "1. VALIDATORS: Hard logical invariants/constraints (e.g., 'X must never happen').\n"
            "2. SOLVERS: Procedures or steps to solve a specific domain problem (e.g., 'To fix Y, do Z').\n"
            "3. ADVISORS: Heuristic advice or scripts to heal a violation (e.g., 'If A fails, check B').\n\n"
            "Format your response as a JSON list of objects:\n"
            '[{"name": "string", "type": "validator|solver|advisor", "description": "precise requirement", "healing_hint": "optional prompt for the solver logic"}]\n\n'
            f"Text content:\n{text[:5000]}"
        )
        res = await self.llm.generate(
            prompt=prompt,
            system="Extract structured domain knowledge as JSON.",
            stream=False,
        )
        # Basic JSON extraction
        try:
            import json

            match = re.search(r"(\[.*\])", res, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return []
        except Exception as e:
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
        res = await self.llm.generate(
            prompt=prompt,
            system="Generate Python code blocks only. Block 1 is the Axiom, Block 2 is the optional Healing Script.",
            stream=False,
        )

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
        test_code = await self.llm.generate(
            prompt=test_prompt, system="Generate test code only.", stream=False
        )

        repl = PythonREPL()
        pure_code = code.replace("```python", "").replace("```", "").strip()
        pure_test = test_code.replace("```python", "").replace("```", "").strip()

        # Inject Mocks and Path for Validation
        import asyncio
        import os
        import sys

        # Calculate repo root for REPL injection
        # dream.py is located at: .../graph_rlm/backend/src/core/dream.py
        # We need to add the repo root (parent of graph_rlm package) to sys.path

        current_file = os.path.abspath(__file__)
        core_dir = os.path.dirname(current_file)  # .../src/core
        src_dir = os.path.dirname(core_dir)  # .../src
        backend_dir = os.path.dirname(src_dir)  # .../backend
        pkg_dir = os.path.dirname(backend_dir)  # .../graph_rlm (package)
        repo_root = os.path.dirname(pkg_dir)  # .../ (repo root)

        import numpy as np

        class UniversalAsyncMock:
            def __getattr__(self, name):
                return UniversalAsyncMock()

            def __call__(self, *args, **kwargs):
                async def _dummy():
                    return "MOCK_RESULT"

                return _dummy()

        repl.namespace.update(
            {
                "asyncio": asyncio,
                "np": np,
                "sys": sys,
                "session_id": "dreamer_validation_session",
                "rlm": UniversalAsyncMock(),
                "mcp": UniversalAsyncMock(),
                "_repo_root": repo_root,
            }
        )

        # Pre-execution: Add repo root to sys.path inside the REPL
        await repl.execute(
            "import sys; sys.path.insert(0, _repo_root) if _repo_root not in sys.path else None"
        )

        logger.info("Executing axiom code...")
        _, stderr, result_axiom, is_err = await repl.execute(pure_code)
        if is_err:
            logger.error("Axiom code execution failed: %s", stderr)
            return False

        logger.info("Executing test code...")
        _, stderr_test, result_test, is_err_test = await repl.execute(pure_test)
        if is_err_test:
            logger.error("Test code execution failed: %s", stderr_test)
        else:
            logger.info("Test code executed successfully.")

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
        from ..mcp_integration.skills import get_axioms_manager

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

    async def _classify_domain(self, insight: str) -> str:
        prompt = (
            f"Classify the technical domain of the following engineering insight.\n"
            f"Examples: 'Arithmetic', 'Network', 'Database', 'Validation', 'Security'.\n"
            f"Output ONE word only.\n\n"
            f"Insight: {insight[:5000]}"
        )
        domain = await self.llm.generate(
            prompt=prompt, system="Output a single CamelCase domain name.", stream=False
        )
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
            except Exception as e:
                logger.warning(
                    "Failed to codify knowledge item '%s...': %s",
                    item.get("name", "unknown"),
                    e,
                )
        return {"status": "success", "codified_axioms": codified}

    async def _get_node_scan_async(self, node_id: str) -> Dict[str, Any]:
        """Async version of node scan."""
        try:
            # db.query is usually sync, but we can loop it in the caller if needed
            # Here we just wrap the existing call
            return self._get_node_scan(node_id)
        except Exception:
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
        except Exception as e:
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
        cypher = "CREATE (i:Insight {id: $id, content: $content, created_at: timestamp(), type: 'dream_consolidation'})"
        self.db.query(cypher, {"id": insight_id, "content": content})
        logger.info(
            "Insight %s saved to graph (not appending to rules.md per RALPH methodology)",
            insight_id[:8],
        )


dreamer = Dreamer()
