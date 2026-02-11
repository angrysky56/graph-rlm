"""
Scratchpad Builder for Stateless Agent Context

Constructs a compact, actionable scratchpad for the agent that includes:
- Current datetime
- Session and REPL metadata with FalkorDB timestamps
- Per-step progress with results and next actions
- Active sub-REPL status

Raw code and full outputs are SAVED in the graph and accessible via rlm.recall(repl_id),
but NOT included in immediate context to prevent bloat.
"""

from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, List

from .db import GraphClient, db
from .logger import get_logger

logger = get_logger("graph_rlm.scratchpad_builder")


class ScratchpadBuilder:
    """Builds a structured scratchpad for the stateless agent."""

    def __init__(self):
        self.db: GraphClient = db

    def build_scratchpad(
        self,
        session_id: str,
        root_session_id: str,
        task: str,
        current_step: int = 0,
        max_steps: int = 1000,
        current_round_id: str = "",
    ) -> str:
        """
        Build a complete scratchpad for the agent.

        Uses round-based architecture:
        - Previous rounds: Compressed summaries with REPL ID pointers
        - Current round: Full detail of progress

        Args:
            session_id: Current REPL session ID
            root_session_id: Root session ID (for sub-REPL tracking)
            task: The current task/prompt
            current_step: Current step number
            max_steps: Maximum allowed steps
            current_round_id: ID of the current round (for filtering)
        """
        lines = []

        # === Header with current datetime ===
        now = datetime.now(timezone.utc)
        local_now = datetime.now()

        # Count completed rounds for round number display
        completed_rounds = self.db.get_completed_rounds(root_session_id)
        current_round_num = len(completed_rounds) + 1

        lines.append("## Agent Session State")
        lines.append(
            f"- **Current Time (UTC)**: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )
        lines.append(
            f"- **Current Time (Local)**: {local_now.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append(f"- **Session ID**: `{session_id}`")
        lines.append(f"- **Round**: {current_round_num} / ∞")
        lines.append(f"- **Step**: {current_step} / {max_steps}")
        lines.append("")

        # === Previous Rounds (COMPRESSED) ===
        if completed_rounds:
            lines.append("## Previous Rounds (Compressed)")
            lines.append("| # | Round ID | Prompt | REPLs | Result |")
            lines.append("|---|----------|--------|-------|--------|")
            for i, r in enumerate(completed_rounds, 1):
                rid = r.get("round_id") or "unknown"
                prompt = (r.get("user_prompt") or "")[:2000]
                if len(r.get("user_prompt") or "") > 2000:
                    prompt += "..."
                repl_ids = r.get("repl_ids") or []
                repls = ", ".join(repl_ids[:2])
                if len(repl_ids) > 2:
                    repls += f" +{len(repl_ids)-2}"
                result = (r.get("final_response") or "")[:5000]
                if len(r.get("final_response") or "") > 5000:
                    result += "..."
                lines.append(f"| {i} | `{rid[:8]}` | {prompt} | {repls} | {result} |")
            lines.append(
                "*(Use `await rlm.recall('ID')` with Round, REPL, or Node IDs for full context)*"
            )
            lines.append("")

        # === Logical State Audit (Failures & Resolutions) ===
        audit_section = self._build_logical_audit(session_id, root_session_id)
        if audit_section:
            lines.append("## Logical State Audit (Knots)")
            lines.append(audit_section)
            lines.append("")

        # === Graph Topology Context (Recent Nodes) ===
        # CRITICAL FIX: Give Dreamer visibility into immediate neighbors
        recent_nodes = self.db.query(
            """
            MATCH (t:Thought {session_id: $sid})
            RETURN t.id as id, t.status as status, t.execution_summary as summary
            ORDER BY t.created_at DESC LIMIT 3
            """,
            {"sid": session_id},
        )
        if recent_nodes:
            lines.append("## Immediate Graph Context")
            for node in recent_nodes:
                summary = node.get("summary") or "No summary"
                lines.append(f"- [{node['id']}] ({node['status']}): {summary}")
            lines.append("")

        # === Current Task ===
        lines.append("## Current Task")
        # Prepend a tag to avoid leading slashes triggering LLM command parsers (Gemini 3)
        lines.append(f"Task: {task}")
        lines.append("")

        # === Current Round Progress (DETAILED) ===
        progress_section = self._build_current_round_progress(
            session_id, root_session_id, current_round_id
        )
        if progress_section:
            lines.append("## Execution Trace (Current Round)")
            lines.append(progress_section)
            lines.append("")

        # === Active Sub-REPLs === These need ID's these are the turns. Turns MUST be tracked and saved
        sub_repls = self._get_sub_repls(root_session_id, session_id)
        if sub_repls:
            lines.append("## Active Sub-REPLs")
            for repl in sub_repls:
                lines.append(repl)
            lines.append("")

        # === Recall Instructions ===
        lines.append("## Data Recall")
        lines.append(
            "- Full code/output saved in graph. Use `await rlm.recall('ID')` (Round, REPL, or Node ID)."
        )
        lines.append(
            "- Use `await rlm.search(query)` for semantic search across all sessions."
        )

        # === Task Completion Gate ===
        # This prevents "exploration loop without synthesis" by showing what phase we're in
        gate_section = self._build_task_completion_gate(session_id, root_session_id)
        if gate_section:
            lines.append("")
            lines.append(gate_section)

        return "\n".join(lines)

    def _build_current_round_progress(
        self, session_id: str, root_session_id: str, _current_round_id: str
    ) -> str:
        """
        Build progress for ONLY the current round (not archived rounds).
        Filters by session_id to maintain recursive branch isolation.
        """
        try:
            # Get ALL thoughts for current session to prevent "orphan round" context gaps
            # We filter by session_id, regardless of round_id, to catch pre-restart thoughts.
            q = """
            MATCH (n:Thought)
            WHERE n.root_session_id = $rsid
            AND n.session_id = $sid
            RETURN n.id as id,
                   n.prompt as prompt,
                   n.status as status,
                   n.result as result,
                   n.created_at as created_at,
                   n.repl_id as repl_id,
                   n.execution_summary as execution_summary,
                   n.next_action as next_action,
                   n.dreamer_analysis as dreamer_analysis,
                   n.final_response as final_response,
                   n.turn_id as turn_id,
                   n.step_id as step_id,
                   n.code_hash as code_hash
            ORDER BY n.turn_id ASC, n.step_id ASC, n.created_at ASC
            """
            results = self.db.query(
                q,
                {"rsid": root_session_id, "sid": session_id},
            )

            if not results:
                return "No progress recorded yet."

            return self._format_progress_rows(results)

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to build current round progress: %s", e)
            return f"Error loading progress: {e}"

    def _format_progress_rows(self, results: List[Any]) -> str:
        """
        Format progress rows as lean summaries with REPL pointers.
        Each step shows: status, step #, timestamp, action type, summary (~150 chars), REPL ID.
        Full code/results live in the REPL/DB — agent can query if needed.
        """
        lines = []
        processed_data = []

        # 1. Normalize data
        for row in results:
            if not row:
                continue
            if isinstance(row, dict):
                processed_data.append(row)
            else:
                # Indices match query order in _build_current_round_progress
                processed_data.append(
                    {
                        "id": row[0],
                        "prompt": row[1],
                        "status": row[2],
                        "result": row[3],
                        "created_at": row[4],
                        "repl_id": row[5],
                        "execution_summary": row[6] or row[3],
                        "next_action": row[7],
                        "dreamer_analysis": row[8],
                        "final_response": row[9],
                        "turn_id": row[10],
                        "step_id": row[11],
                        "code_hash": row[12],
                    }
                )

        # 2. Group by Turn and Format
        i = 0
        current_turn = None
        while i < len(processed_data):
            row = processed_data[i]
            turn_id = row.get("turn_id")
            repl_id = row.get("repl_id") or "unknown"

            # Check for Turn boundary
            if turn_id != current_turn:
                current_turn = turn_id
                turn_label = (
                    f"Turn {turn_id}" if turn_id is not None else "Initial Setup"
                )
                lines.append(f"\n### 🛠️ {turn_label} [Environment: **{repl_id}**]")
                lines.append("---")

            status = row.get("status") or "unknown"
            dreamer_analysis = (row.get("dreamer_analysis") or "").strip()

            # Condensation Logic: Group consecutive rejections with same analysis
            # (Keeping this but making it within the turn)
            if status == "rejected" and dreamer_analysis:
                group_start_idx = i
                while (
                    i + 1 < len(processed_data)
                    and processed_data[i + 1].get("status") == "rejected"
                    and (processed_data[i + 1].get("dreamer_analysis") or "").strip()
                    == dreamer_analysis
                    and processed_data[i + 1].get("turn_id") == turn_id
                ):
                    i += 1
                group_count = i - group_start_idx + 1

                if group_count > 1:
                    lines.append(
                        f"FAIL Step {group_start_idx+1}-{i+1} (DREAMER): "
                        f"REJECTED {group_count} TIMES for same pattern: {dreamer_analysis}"
                    )
                    i += 1
                    continue

            # === LEAN SUMMARY FORMATTING ===
            # Goal: ~150 char summary per step with REPL pointer for stateless continuity
            prompt = row.get("prompt") or ""
            status_sym = {
                "success": "✓",
                "running": "⏳",
                "pending": "⋯",
                "failed": "✗",
                "error": "⚠",
                "reflexion": "🔄",
                "rejected": "🚫",
            }.get(status, "?")

            ts_str = ""
            created_at = row.get("created_at")
            if created_at:
                with suppress(Exception):
                    ts = datetime.fromtimestamp(created_at / 1000)
                    ts_str = ts.strftime("%H:%M:%S")

            # Intent classification
            clean_prompt = prompt.strip()
            if any(k in clean_prompt for k in ["def ", "import ", "await ", "="]):
                action_type = "Code"
            elif any(k in clean_prompt for k in ["REFLEXION", "INTERVENTION"]):
                action_type = "System"
            else:
                action_type = "Thought"

            # Build concise summary (flattened, no truncation)
            summary = clean_prompt.replace("\n", " ").strip()

            # For Code: add brief result indicator
            exec_summary = row.get("execution_summary") or row.get("result") or ""
            if action_type == "Code" and exec_summary:
                result_preview = str(exec_summary).strip().replace("\n", " ")
                if result_preview:
                    summary += f" → {result_preview}"

            step_num = row.get("step_id") if row.get("step_id") is not None else i + 1
            code_hash = row.get("code_hash")
            hash_str = f"#{code_hash[:6]}" if code_hash else ""

            # Compact format: ✓ S3 12:34:56 [Code] "Installing numpy" → OK | REPL:abc123 #deadbe
            step_line = f"{status_sym} S{step_num} {ts_str} [{action_type}] {summary}"
            step_line += f" | REPL:{repl_id}"
            if hash_str:
                step_line += f" {hash_str}"

            # CRITICAL: Dreamer mandates MUST be visible for self-healing
            if dreamer_analysis:
                if status in ["rejected", "reflexion"]:
                    step_line += f"\n    🚨 MANDATE: {dreamer_analysis[:500]}"
                else:
                    step_line += f"\n    💭 {dreamer_analysis[:100]}"

            lines.append(step_line)
            i += 1

        return "\n".join(lines) if lines else "No progress recorded yet."

    def _get_sub_repls(
        self, root_session_id: str, current_session_id: str
    ) -> List[str]:
        """Get active sub-REPL sessions with their status."""
        try:
            q = """
            MATCH (n:Thought)
            WHERE n.root_session_id = $root_id AND n.session_id <> $current_id
            WITH n.session_id as sid,
                 count(n) as thought_count,
                 collect(n.prompt)[0] as initial_prompt,
                 max(n.created_at) as last_activity,
                 collect(n.status)[-1] as last_status,
                 collect(n.result)[-1] as last_result,
                 collect(n.prompt)[-1] as last_action
            RETURN sid, thought_count, initial_prompt, last_activity, last_status, last_result, last_action
            ORDER BY last_activity DESC
            LIMIT 10
            """
            results = self.db.query(
                q, {"root_id": root_session_id, "current_id": current_session_id}
            )

            lines = []
            for row in results:
                if isinstance(row, dict):
                    sid = row.get("sid", "unknown")
                    # count = row.get("thought_count", 0) # Unused
                    prompt = row.get("initial_prompt") or ""
                    status = row.get("last_status", "unknown")
                    last_activity = row.get("last_activity", "")
                    last_res = (
                        row.get("last_result") or ""
                    )  # Fetch full, truncate later
                    last_act = row.get("last_action") or ""
                else:
                    sid = row[0]
                    # count = row[1] # Unused
                    prompt = row[2] or ""
                    status = row[4] if len(row) > 4 else "unknown"
                    last_activity = row[3] if len(row) > 3 else ""
                    last_res = row[5] if len(row) > 5 else ""
                    last_act = row[6] if len(row) > 6 else ""

                # Format timestamp
                ts_str = ""
                if last_activity:
                    try:
                        if isinstance(last_activity, (int, float)):
                            ts = datetime.fromtimestamp(last_activity / 1000)
                            ts_str = ts.strftime("%H:%M:%S")
                    except Exception as e:  # pylint: disable=broad-except
                        logger.warning("Failed to format sub-REPL timestamp: %s", e)

                status_sym = (
                    "🟢"
                    if status == "success"
                    else "🔵" if status == "running" else "🔴"
                )

                # Truncate for display (but keep enough for context)
                safe_prompt = prompt[:200] + "..." if len(prompt) > 200 else prompt
                safe_act = last_act[:200] + "..." if len(last_act) > 200 else last_act
                safe_res = last_res[:200] + "..." if len(last_res) > 200 else last_res

                # Structured Block for Visibility
                block = (
                    f"### 🔄 Sub-REPL [ID: {sid}] ({status_sym} {status.upper()})\n"
                    f"**Time**: {ts_str}\n"
                    f"**Task**: {safe_prompt}\n"
                    f"**Last Action**: {safe_act}\n"
                    f"**Last Result**: {safe_res}\n"
                    f"_(Use `await rlm.recall('{sid}')` to inspect details)_"
                )
                lines.append(block)

            return lines

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to get sub-REPLs: %s", e)
            return []

    def _build_task_completion_gate(self, session_id: str, root_session_id: str) -> str:
        """
        Task Completion Gate: Prevents "exploration loop without synthesis".

        Shows the agent:
        1. How many REPL operations have been completed (exploration phase)
        2. Whether a final deliverable/synthesis exists
        3. Clear directive: if exploration done, proceed to synthesis (don't re-explore)
        """
        try:
            lines = ["## 🚦 Task Completion Status"]

            # 1. Count successful REPL operations (exploration work done)
            q_repl_ops = """
            MATCH (n:Thought)
            WHERE (n.root_session_id = $rsid OR n.session_id = $sid)
            AND n.prompt IS NOT NULL
            AND (n.status = 'success' OR n.status = 'completed' OR n.result IS NOT NULL)
            RETURN count(n) as op_count
            """
            repl_result = self.db.query(
                q_repl_ops, {"rsid": root_session_id, "sid": session_id}
            )
            op_count = 0
            if repl_result and len(repl_result) > 0:
                if isinstance(repl_result[0], dict):
                    op_count = repl_result[0].get("op_count", 0)
                else:
                    op_count = repl_result[0][0] if repl_result[0] else 0

            # 2. Check for read/exploration actions specifically
            # Use prompt and execution_summary which are the actual node properties
            q_exploration = """
            MATCH (n:Thought)
            WHERE (n.root_session_id = $rsid OR n.session_id = $sid)
            AND (
                (n.prompt IS NOT NULL AND (n.prompt CONTAINS 'read' OR n.prompt CONTAINS 'list' OR n.prompt CONTAINS 'explore' OR n.prompt CONTAINS 'README'))
                OR (n.execution_summary IS NOT NULL AND (n.execution_summary CONTAINS 'read' OR n.execution_summary CONTAINS 'explored'))
                OR (n.result IS NOT NULL AND n.result CONTAINS 'bytes')
            )
            AND (n.status = 'success' OR n.status = 'completed' OR n.result IS NOT NULL)
            RETURN count(n) as explore_count
            """
            explore_result = self.db.query(
                q_exploration, {"rsid": root_session_id, "sid": session_id}
            )
            explore_count = 0
            if explore_result and len(explore_result) > 0:
                if isinstance(explore_result[0], dict):
                    explore_count = explore_result[0].get("explore_count", 0)
                else:
                    explore_count = explore_result[0][0] if explore_result[0] else 0

            # 3. Check for synthesis/report actions
            # Use prompt and execution_summary which are the actual node properties
            q_synthesis = """
            MATCH (n:Thought)
            WHERE (n.root_session_id = $rsid OR n.session_id = $sid)
            AND (
                (n.prompt IS NOT NULL AND (n.prompt CONTAINS 'report' OR n.prompt CONTAINS 'synthesize' OR n.prompt CONTAINS 'compare' OR n.prompt CONTAINS 'analyze' OR n.prompt CONTAINS 'summary'))
                OR (n.execution_summary IS NOT NULL AND (n.execution_summary CONTAINS 'report' OR n.execution_summary CONTAINS 'comparison' OR n.execution_summary CONTAINS 'analysis'))
            )
            AND (n.status = 'success' OR n.status = 'completed' OR n.result IS NOT NULL)
            RETURN count(n) as synth_count
            """
            synth_result = self.db.query(
                q_synthesis, {"rsid": root_session_id, "sid": session_id}
            )
            synth_count = 0
            if synth_result and len(synth_result) > 0:
                if isinstance(synth_result[0], dict):
                    synth_count = synth_result[0].get("synth_count", 0)
                else:
                    synth_count = synth_result[0][0] if synth_result[0] else 0

            # 4. Determine phase and provide directive
            lines.append(f"- **REPL Operations Completed**: {op_count}")
            lines.append(f"- **Exploration Actions**: {explore_count}")
            lines.append(f"- **Synthesis Actions**: {synth_count}")

            if explore_count > 0 and synth_count == 0:
                # Exploration done, synthesis pending
                lines.append("")
                lines.append("> [!IMPORTANT]")
                lines.append("> **Phase: SYNTHESIS PENDING**")
                lines.append(
                    "> Exploration is complete. DO NOT re-read files or re-explore."
                )
                lines.append(
                    "> Proceed directly to SYNTHESIS: generate the report/analysis/comparison."
                )
                lines.append(
                    "> Use `await rlm.recall('ID')` to retrieve previously read content."
                )
            elif synth_count > 0:
                # Synthesis done
                lines.append("")
                lines.append("> [!TIP]")
                lines.append("> **Phase: TASK COMPLETE**")
                lines.append("> Both exploration and synthesis have been performed.")
            elif op_count == 0:
                # No work done yet
                lines.append("")
                lines.append("> [!NOTE]")
                lines.append("> **Phase: EXPLORATION NEEDED**")
                lines.append(
                    "> No REPL operations detected. Start by reading/exploring the required files."
                )

            return "\n".join(lines)

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to build task completion gate: %s", e)
            return ""

    def _build_logical_audit(self, session_id: str, root_session_id: str) -> str:
        """
        Builds a summary of Active vs Resolved failure knots.
        This prevents the 'Groundhog Day' loop by showing the agent what is already fixed.
        """
        try:
            lines = []

            # 1. Active Knots (Failures needing attention)
            q_active = """
            MATCH (n:Thought)
            WHERE (n.root_session_id = $rsid OR n.session_id = $sid)
            AND (n.status = 'failed' OR n.status = 'error')
            AND (n.dreamer_checked IS NULL OR n.dreamer_checked = false)
            RETURN n.id, n.prompt, n.result
            ORDER BY n.created_at DESC LIMIT 3
            """
            active_res = self.db.query(
                q_active, {"rsid": root_session_id, "sid": session_id}
            )

            if active_res:
                lines.append("### 🔴 Active Failure Knots (Needs Fix)")
                for row in active_res:
                    rid, prompt, res = (
                        row
                        if isinstance(row, list)
                        else (row["n.id"], row["n.prompt"], row["n.result"])
                    )
                    # Safe extraction
                    if isinstance(row, dict):
                        rid = row.get("n.id") or row.get("id")
                        prompt = row.get("n.prompt") or row.get("prompt")
                        res = row.get("n.result") or row.get("result")

                    short_p = (prompt or "")[:200]
                    short_r = (res or "")[:500]
                    lines.append(f"- [!] `{rid}`: {short_p} -> {short_r}")
            else:
                lines.append("### 🟢 Logical State Clean (No Active Failures)")

            # 2. Recently Resolved Knots (Failures marked checked or consolidated)
            # We look for consolidated/checked failures in this session/root to show progress
            q_resolved = """
            MATCH (n:Thought)
            WHERE (n.root_session_id = $rsid OR n.session_id = $sid)
            AND (n.status = 'consolidated' OR ((n.status='failed' OR n.status='error') AND n.dreamer_checked = true))
            RETURN n.id, n.prompt
            ORDER BY n.created_at DESC LIMIT 3
            """
            resolved_res = self.db.query(
                q_resolved, {"rsid": root_session_id, "sid": session_id}
            )

            if resolved_res:
                lines.append("")
                lines.append("### ✅ Recently Resolved (Dreamer Acknowledged)")
                for row in resolved_res:
                    if isinstance(row, dict):
                        prompt = row.get("n.prompt") or row.get("prompt")
                    else:
                        prompt = row[1]
                    lines.append(f"- [x] {(prompt or '')[:60]}...")

            return "\n".join(lines)

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to build logical audit: %s", e)
            return ""


# Singleton instance
scratchpad_builder = ScratchpadBuilder()
