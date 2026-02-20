"""
Scratchpad Builder for Stateless Agent Context

Constructs a compact, actionable scratchpad for the agent that includes:
- Current datetime
- Session and REPL metadata with FalkorDB timestamps
- Per-step progress with results and next actions (System TUI Table)
- Active sub-REPL status

Raw code and full outputs are SAVED in the graph and accessible via rlm.recall(repl_id),
but NOT included in immediate context to prevent bloat.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from .circuit import generate_correlation_id, get_correlation_id
from .config import settings
from .db import GraphClient, db
from .logger import get_logger
from .services.circuit import protected_llm_generate

logger = get_logger("graph_rlm.scratchpad_builder")


class ScratchpadBuilder:
    """Builds a structured scratchpad for the stateless agent."""

    def __init__(self):
        self.db: GraphClient = db

    async def build_scratchpad(
        self,
        session_id: str,
        root_session_id: str,
        task: str,
        current_step: int = 0,
        max_steps: int = 1000,
        current_round_id: str = "",
        morph_gestalt: Optional[str] = None,
        current_repl_id: Optional[str] = None,
    ) -> str:
        """
        Build a complete scratchpad for the agent.

        Uses round-based architecture:
        - Previous rounds: Compressed summaries (Block Format)
        - Current round: Full detail of progress (System TUI Table) + Observability
        - Thimac Gestalt: Existence/Subsistence overview
        """
        lines = []

        # === Header with current datetime ===
        now = datetime.now(timezone.utc)

        # Count completed rounds for round number display
        completed_rounds = self.db.get_completed_rounds(root_session_id)
        current_round_num = len(completed_rounds) + 1

        lines.append("## Agent Session State (System TUI)")
        lines.append(f"- **Time**: {now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} UTC")
        header = f"Session: {session_id[:8]}... | Round: {current_round_num} | Step: {current_step}/{max_steps}"
        if current_repl_id:
            header += f" | Active REPL: {current_repl_id}"

        # Fetch Last Successful REPL for grounding
        q_last_success = """
        MATCH (n:Thought)
        WHERE (n.root_session_id = $rsid OR n.session_id = $sid)
        AND n.status IN ['completed', 'success']
        AND n.repl_id IS NOT NULL
        RETURN n.repl_id
        ORDER BY n.created_at DESC LIMIT 1
        """
        last_repl_res = self.db.query(
            q_last_success, {"rsid": root_session_id, "sid": session_id}
        )
        if last_repl_res:
            last_repl = (
                last_repl_res[0].get("n.repl_id") or last_repl_res[0].get("repl_id")
                if isinstance(last_repl_res[0], dict)
                else last_repl_res[0][0]
            )
            header += f" | Last Successful REPL: {last_repl}"

        header += "\n" + "=" * 60 + "\n"
        lines.append(header)
        lines.append("")

        # === Thimac Memory Gestalt (Primary Anchor) ===
        if morph_gestalt:
            lines.append("## Thimac Gestalt (Memory Anchor)")
            lines.append(morph_gestalt)
            lines.append("")

        # === Previous Rounds (Variable Length Summaries) ===
        if completed_rounds:
            lines.append("## Previous Rounds (Compressed Context)")

            # Prepare summarization tasks
            summary_tasks = []
            for r in completed_rounds:
                prompt_text = r.get("user_prompt") or ""
                result_text = r.get("final_response") or ""

                summary_tasks.append(
                    self._summarize_content(prompt_text, "User Prompt")
                )
                summary_tasks.append(
                    self._summarize_content(result_text, "Agent Result")
                )

            # Execute all summaries in parallel
            summaries = await asyncio.gather(*summary_tasks)

            # Re-assemble
            summary_idx = 0
            for i, r in enumerate(completed_rounds, 1):
                rid = r.get("round_id") or "unknown"
                repl_ids = r.get("repl_ids") or []
                repls_str = ", ".join(repl_ids) if repl_ids else "None"

                prompt_summary = summaries[summary_idx]
                result_summary = summaries[summary_idx + 1]
                summary_idx += 2

                lines.append(f"### Round {i} (ID: `{rid}`)")
                lines.append("**User Prompt**:")
                lines.append(prompt_summary)
                lines.append("")
                lines.append("**Agent Result**:")
                lines.append(result_summary)
                lines.append("")
                lines.append(f"**REPLs**: `{repls_str}`")
                lines.append("")

        # === Current Round Progress (The Trace) ===
        progress = await self._build_current_round_progress(
            session_id, root_session_id, current_round_id
        )
        lines.append("## Execution Trace (Current Round)")
        lines.append(progress)
        lines.append("")

        # === Logical State Audit (Failures & Resolutions) ===
        audit_section = self._build_logical_audit(session_id, root_session_id)
        if audit_section:
            lines.append("## Logical State Audit (Knots)")
            lines.append(audit_section)
            lines.append("")

        # === Missing Requirements (Cognitive Gap) ===
        if morph_gestalt:
            missing = self._build_missing_requirements(task, morph_gestalt)
            if missing:
                lines.append("## Missing Requirements (The Gap)")
                lines.append(missing)
                lines.append("")

        # === Recall Instructions ===
        lines.append("## Data Commands")
        lines.append(
            "- `await rlm.recall('ID')`: Retrieve full node content (Code/Result)"
        )
        lines.append(
            "- `await rlm.search('query')`: Semantic search across FULL history (including truncated steps)"
        )

        return "\n".join(lines)

    async def _summarize_content(self, text: str, label: str) -> str:
        """
        Summarize content using the LLM if it exceeds the threshold.
        Returns detailed summary or raw text.
        """
        if not text:
            return "(empty)"

        # Threshold: Don't waste LLM calls on short text
        if len(text) < 2000:
            return text.strip()

        try:
            summary_model = settings.SUMMARY_MODEL or "gemini-2.0-flash-lite"

            prompt = (
                f"You are a summarizer for a stateless agent's memory. "
                f"Summarize the following {label}.\n"
                "Retain ALL critical constraints, code snippets, specific values, and errors.\n"
                "Remove fluff but maximize informational density.\n"
                f"---\n{text[:8000]}\n---"
            )

            summary = await protected_llm_generate(
                prompt,
                model=summary_model,
                correlation_id=get_correlation_id() or generate_correlation_id(),
            )
            return summary.strip()
        except (httpx.RequestError, ValueError, RuntimeError) as e:
            logger.warning("Summary generation failed: %s", e)
            return text[:500] + "... [Truncated due to summary error]"

    async def _build_current_round_progress(
        self, session_id: str, root_session_id: str, current_round_id: str
    ) -> str:
        """
        Build progress for ONLY the current round (not archived rounds).
        """
        try:
            # Get ALL thoughts for current session with Observability Fields
            q = """
            MATCH (n:Thought)
            WHERE n.root_session_id = $rsid
            AND n.session_id = $sid
            AND n.round_id = $rid
            RETURN n.id as id,
                   n.prompt as prompt,
                   n.status as status,
                   n.result as result,
                   n.created_at as created_at,
                   n.repl_id as repl_id,
                   n.turn_id as turn_id,
                   n.step_id as step_id,
                   n.sheaf_score as sheaf_score,
                   n.spectral_energy as spectral_energy,
                   n.dreamer_analysis as dreamer_analysis,
                   n.execution_summary as execution_summary,
                   n.repe_shakiness as repe_shakiness,
                   n.repe_evasion as repe_evasion,
                   n.omcd_score as omcd_score
            ORDER BY n.turn_id ASC, n.step_id ASC, n.created_at ASC
            LIMIT 2000
            """
            results = self.db.query(
                q,
                {"rsid": root_session_id, "sid": session_id, "rid": current_round_id},
            )

            if not results:
                return "No progress recorded yet."

            return await self._format_progress_rows(results)

        except (AttributeError, RuntimeError, ValueError) as e:
            logger.error("Failed to build current round progress: %s", e)
            return f"Error loading progress: {e}"

    async def _format_progress_rows(self, results: List[Any]) -> str:
        """
        Format progress as a clean System TUI table with Observability Alerts.
        """
        processed_data = []
        for row in results:
            if not row:
                continue
            if isinstance(row, dict):
                processed_data.append(row)
            else:
                # Handle list-based return (raw driver)
                # Ensure we handle the new columns safely even if DB returns fewer
                base_data = {
                    "id": row[0],
                    "prompt": row[1],
                    "status": row[2],
                    "result": row[3],
                    "created_at": row[4],
                    "repl_id": row[5],
                    "turn_id": row[6],
                    "step_id": row[7],
                    "sheaf_score": row[8],
                    "spectral_energy": row[9],
                    "dreamer_analysis": row[10],
                    "execution_summary": row[11],
                }
                # Check bounds for new columns
                if len(row) > 12:
                    base_data["repe_shakiness"] = row[12]
                if len(row) > 13:
                    base_data["repe_evasion"] = row[13]
                if len(row) > 14:
                    base_data["omcd_score"] = row[14]

                processed_data.append(base_data)

        if not processed_data:
            return "No progress rows."

        # [WINDOWING] Disabled - Provide full trace for maximum fidelity (1M token budget)
        windowed_data = processed_data

        # Prepare batch LLM tasks for step summaries
        summary_tasks = []
        for row in windowed_data:
            summary_tasks.append(self._generate_step_summary(row))

        # Parallel execution
        summaries = await asyncio.gather(*summary_tasks)

        # Build Table with Row Collapsing (Deduplication)
        lines = []
        lines.append(
            "| Time | REPL | T.S | St | Summary (Auto-Generated) | Recall ID |"
        )
        lines.append("|---|---|---|---|---|---|")

        skip_until = -1
        for idx, row in enumerate(windowed_data):
            if idx <= skip_until:
                continue

            # Check for consecutive identical summaries
            current_summary = summaries[idx]
            match_count = 1
            last_idx = idx

            for look_idx in range(idx + 1, len(windowed_data)):
                # ONLY collapse if turn_id is the same AND summary is the same.
                # Never collapse across different turns, as that confuses the Dreamer.
                if summaries[look_idx] == current_summary and windowed_data[
                    look_idx
                ].get("turn_id") == row.get("turn_id"):
                    match_count += 1
                    last_idx = look_idx
                else:
                    break

            if match_count >= 3:
                # Collapse Rows
                ts_start = row.get("step_id", "?")
                ts_end = windowed_data[last_idx].get("step_id", "?")
                turn = row.get("turn_id", "?")

                lines.append(
                    f"| --:--:-- | `SYSTEM` | {turn}.{ts_start}-{ts_end} | 🔄 | "
                    f"**REPETITIVE ACTION**: {current_summary} repeated {match_count} times (No new data) | `N/A` |"
                )
                skip_until = last_idx
                continue

            # Time
            ts_str = "--:--:--"
            if row.get("created_at"):
                dt = datetime.fromtimestamp(row["created_at"] / 1000)
                ts_str = dt.strftime("%H:%M:%S")

            # REPL
            repl = (row.get("repl_id") or "       ")[:7]

            # T.S
            turn = row.get("turn_id")
            step = row.get("step_id")
            ts_display = f"{turn}.{step}" if turn is not None else "??"

            # Status
            st_map = {
                "success": "✓",
                "failed": "✗",
                "rejected": "🛡️",
                "running": "⏳",
                "pending": "⋯",
                "fragment": "🧩",
                "valid": "✅",
                "navigator": "🧭",
                "omcd": "⚖️",
                "sheaf": "📐",
                "repe": "Ψ",
                "reflexion": "🧠",
            }
            st = st_map.get(row.get("status"), "?")

            # Summary
            summary = summaries[idx]

            # --- OBSERVABILITY ALERTS ---
            alerts = []

            # Sheaf Loop Detection
            sheaf = row.get("sheaf_score")
            if sheaf is not None and sheaf > 0.7:
                alerts.append(f"> [!] SHEAF: Loop Detected ({sheaf:.2f})")

            # Spectral Energy (Drift)
            energy = row.get("spectral_energy")
            if energy is not None and energy > 0.5:
                alerts.append(f"> [!] DRIFT: High Deviation ({energy:.2f})")

            # RepE Psychological Alerts
            shaky = row.get("repe_shakiness")
            evasion = row.get("repe_evasion")
            if shaky is not None and shaky < -0.15:
                alerts.append(f"> [Ψ] SHAKINESS: {shaky:.2f} (Uncertain)")
            if evasion is not None and evasion < -0.15:
                alerts.append(f"> [Ψ] EVASION: {evasion:.2f} (Dodging)")

            # oMCD Stopping
            omcd_q = row.get("omcd_score")
            # Only show explicit stop signals or very low confidence
            if omcd_q is not None and omcd_q < 0.3:
                alerts.append(f"> [Ω] LOW STOP CONFIDENCE: {omcd_q:.2f}")

            if alerts:
                # Add line break and alerts to summary column
                # Markdown tables support <br/> for line breaks
                alert_str = "<br/>".join(alerts)
                summary = f"{summary}<br/>**{alert_str}**"

            # Recall
            tid = row.get("id")
            recall_link = f"`recall('{tid}')`"

            lines.append(
                f"| {ts_str} | `{repl}` | {ts_display} | {st} | {summary} | {recall_link} |"
            )

        return "\n".join(lines)

    async def _generate_step_summary(self, row: Dict) -> str:
        """
        Use SUMMARY_MODEL to generate a 60-char summary of the step.
        """
        prompt = row.get("prompt") or ""
        status = row.get("status")

        # [Preserved existing _generate_step_summary logic, just ensuring it handles new fields if needed]
        if status in [
            "navigator",
            "omcd",
            "sheaf",
            "repe",
            "reflexion",
            "system",
            "fragment",
        ]:
            return prompt.strip()[:80]

        # Use execution_summary if available from DB (cheaper)
        if row.get("execution_summary"):
            return str(row.get("execution_summary"))[:80]

        prompt = row.get("prompt") or ""
        result = row.get("result") or ""
        status = row.get("status")

        # Fast path for very short items
        if len(prompt) < 60 and len(result) < 60:
            return prompt.strip()[:60]

        try:
            summary_model = settings.SUMMARY_MODEL or "gemini-2.0-flash-lite"

            llm_prompt = f"""Summarize this agent action in ONE line (max 60 chars).
Focus on WHAT was done and the OUTCOME.
ACTION: {prompt[:1000]}
RESULT: {result[:1000]}
STATUS: {status}

Summary:"""

            summary = await protected_llm_generate(
                llm_prompt,
                model=summary_model,
                correlation_id=get_correlation_id() or generate_correlation_id(),
            )
            return summary.strip()[:80].replace("\n", " ")
        except (httpx.RequestError, ValueError, RuntimeError):
            # Fallback
            return prompt.strip()[:80].replace("\n", " ")

    def _get_sub_repls(
        self, root_session_id: str, current_session_id: str
    ) -> List[str]:
        """Get active sub-REPL sessions with their status."""
        try:
            q = """
            MATCH (n:Thought)
            WHERE n.root_session_id = $root_id AND n.session_id <> $current_id
            WITH n.session_id as sid,
                 max(n.created_at) as last_activity,
                 collect(n.status)[-1] as last_status,
                 collect(n.prompt)[-1] as last_action
            RETURN sid, last_activity, last_status, last_action
            ORDER BY last_activity DESC
            LIMIT 5
            """
            results = self.db.query(
                q, {"root_id": root_session_id, "current_id": current_session_id}
            )

            lines = []
            for row in results:
                if isinstance(row, dict):
                    sid = row.get("sid")
                    status = row.get("last_status")
                    ts = row.get("last_activity")
                else:
                    sid = row[0]
                    ts = row[1]
                    status = row[2]

                # Format
                dt_str = "unknown"
                if ts:
                    dt = datetime.fromtimestamp(ts / 1000)
                    dt_str = dt.strftime("%H:%M:%S")

                sym = "🟢" if status == "success" else "🔴"
                lines.append(f"- {sym} **{sid}** ({dt_str}): {status}")

            return lines
        except (AttributeError, RuntimeError, ValueError) as e:
            logger.error("Failed to get sub-REPLs: %s", e)
            return []

    def _build_logical_audit(self, session_id: str, root_session_id: str) -> str:
        """
        Builds a summary of Active vs Resolved failure knots.
        """
        try:
            q_active = """
            MATCH (n:Thought)
            WHERE (n.root_session_id = $rsid OR n.session_id = $sid)
            AND (n.status = 'failed' OR n.status = 'error')
            RETURN n.id, n.prompt, n.result
            ORDER BY n.created_at DESC LIMIT 3
            """
            active_res = self.db.query(
                q_active, {"rsid": root_session_id, "sid": session_id}
            )

            lines = []
            if active_res:
                lines.append("### 🏛️ Structural Skeleton (Knot Audit)")
                for row in active_res:
                    if isinstance(row, dict):
                        rid = row.get("n.id") or row.get("id")
                        prompt = (
                            row.get("n.prompt") or row.get("prompt") or ""
                        ).lower()
                        result = (
                            row.get("n.result") or row.get("result") or ""
                        ).lower()
                        status = row.get("status") or "failed"
                    else:
                        rid = row[0]
                        prompt = (row[1] or "").lower()
                        result = (row[2] or "").lower()
                        status = "failed"

                    # (S, R, O) Triplet Extraction (Heuristic)
                    # S = Subject (What was being accessed)
                    # R = Relation (The failure type/action)
                    # O = Object (The target component)

                    subject = "Agent"
                    relation = "Failure"
                    target = "Goal"

                    if "error" in result:
                        relation = (
                            result.split(":")[0].strip() if ":" in result else "Error"
                        )
                        # Extract probable subject/object from error message
                        words = result.split()
                        if len(words) > 2:
                            target = words[-1].strip("'\"")

                    rid_display = rid[:8] if rid else "unknown"
                    lines.append(
                        f"- **{status.upper()}** [{rid_display}]: `{subject}` -> `{relation}` -> `{target}`"
                    )
                    # Add H1 Obstruction Warning if energy is high
                    from .sheaf import sheaf

                    h1_score = sheaf.calculate_h1_obstruction(
                        [{"prompt": prompt, "result": result, "status": status}]
                    )
                    if h1_score > 0.5:
                        lines.append(
                            f"  > [!WARNING] High H1 Cohomology Obstruction ({h1_score:.2f}) detected."
                        )

            return "\n".join(lines)
        except (AttributeError, RuntimeError, ValueError):
            return ""

    def _build_missing_requirements(self, task: str, gestalt: str) -> str:
        """
        Heuristic contrast between task and Thimac existence state.
        """
        # Simple scarcity detection
        if "EXISTENCE: No materialized results" in gestalt:
            return "> [!] SCARCITY ALERT: No concrete evidence materialized yet. You are likely stuck in a subsistence loop."

        if ("CREATE" not in gestalt and "RELEASE" not in gestalt) and (
            "implement" in task.lower() or "fix" in task.lower()
        ):
            return "> [!] MISSING ACTION: Task requires implementation, but no CREATE/RELEASE operations detected in history."

        return "> [i] Grounding Check: Ensure current research relates to the specific identifiers in the User Task."


# Singleton instance
scratchpad_builder = ScratchpadBuilder()
