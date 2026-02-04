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
            lines.append("| # | Prompt | REPLs | Result |")
            lines.append("|---|--------|-------|--------|")
            for i, r in enumerate(completed_rounds, 1):
                prompt = (r.get("user_prompt") or "")[:40]
                if len(r.get("user_prompt") or "") > 40:
                    prompt += "..."
                repl_ids = r.get("repl_ids") or []
                repls = ", ".join(repl_ids[:2])
                if len(repl_ids) > 2:
                    repls += f" +{len(repl_ids)-2}"
                result = (r.get("final_response") or "")[:50]
                if len(r.get("final_response") or "") > 50:
                    result += "..."
                lines.append(f"| {i} | {prompt} | {repls} | {result} |")
            lines.append("*(Use `await rlm.recall('repl_id')` for full context)*")
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
                summary = (node.get("summary") or "No summary")[:100]
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

        # === Active Sub-REPLs ===
        sub_repls = self._get_sub_repls(root_session_id, session_id)
        if sub_repls:
            lines.append("## Active Sub-REPLs")
            for repl in sub_repls:
                lines.append(repl)
            lines.append("")

        # === Recall Instructions ===
        lines.append("## Data Recall")
        lines.append(
            "- Full code/output saved in graph. Use `await rlm.recall('{repl_id}')` to retrieve."
        )
        lines.append(
            "- Use `await rlm.search(query)` for semantic search across all sessions."
        )

        return "\n".join(lines)

    def _build_current_round_progress(
        self, session_id: str, root_session_id: str, current_round_id: str
    ) -> str:
        """
        Build progress for ONLY the current round (not archived rounds).

        If current_round_id is empty, get thoughts with no round_id assigned yet.
        """
        try:
            # Get thoughts for current round only
            # Thoughts that have not yet been archived into a Round node
            q = """
            MATCH (n:Thought)
            WHERE n.root_session_id = $rsid
            AND (n.round_id IS NULL OR n.round_id = $crid)
            RETURN n.id as id,
                   n.prompt as prompt,
                   n.status as status,
                   n.result as result,
                   n.created_at as created_at,
                   n.repl_id as repl_id,
                   n.execution_summary as execution_summary,
                   n.next_action as next_action,
                   n.dreamer_analysis as dreamer_analysis,
                   n.final_response as final_response
            ORDER BY n.created_at ASC
            """
            results = self.db.query(
                q, {"rsid": root_session_id, "crid": current_round_id}
            )

            if not results:
                return "No progress recorded yet."

            return self._format_progress_rows(results)

        except Exception as e:
            logger.error(f"Failed to build current round progress: {e}")
            return f"Error loading progress: {e}"

    def _format_progress_rows(self, results: List[Any]) -> str:
        """
        Helper to format detailed progress rows from db results.
        Includes condensation logic for repetitive Dreamer rejections.
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
                    }
                )

        # 2. Group and Format
        i = 0
        while i < len(processed_data):
            row = processed_data[i]
            status = row.get("status") or "unknown"
            dreamer_analysis = (row.get("dreamer_analysis") or "").strip()

            # Condensation Logic: Group consecutive rejections with same analysis
            if status == "rejected" and dreamer_analysis:
                group_start_idx = i
                while (
                    i + 1 < len(processed_data)
                    and processed_data[i + 1].get("status") == "rejected"
                    and (processed_data[i + 1].get("dreamer_analysis") or "").strip()
                    == dreamer_analysis
                ):
                    i += 1
                group_count = i - group_start_idx + 1

                if group_count > 1:
                    lines.append(
                        f"FAIL Step {group_start_idx+1}-{i+1} (DREAMER): "
                        f"REJECTED {group_count} TIMES for same pattern: {dreamer_analysis[:200]}..."
                    )
                    i += 1
                    continue

            # Standard Formatting
            prompt = row.get("prompt") or ""
            status_sym = {
                "success": "DONE",
                "running": "BUSY",
                "pending": "WAIT",
                "failed": "FAIL",
                "error": "ERR",
                "reflexion": "FIX",
                "rejected": "REJECT",
            }.get(status, "???")

            ts_str = ""
            created_at = row.get("created_at")
            if created_at:
                try:
                    ts = datetime.fromtimestamp(created_at / 1000)
                    ts_str = f" ({ts.strftime('%H:%M:%S')})"
                except Exception as e:
                    logger.debug(f"Failed to format timestamp {created_at}: {e}")

            # Intent classification
            clean_prompt = prompt.strip()
            action_type = "Thought"
            summary = clean_prompt

            if any(k in clean_prompt for k in ["def ", "import ", "await "]):
                action_type = "Code"
                prompt_lines = clean_prompt.splitlines()
                summary = prompt_lines[0][:80]
                if len(prompt_lines) > 1:
                    summary += "..."
            elif any(
                k in clean_prompt for k in ["REFLEXION_BREAK", "SYSTEM INTERVENTION"]
            ):
                action_type = "SYSTEM"
                summary = f"⚠️ {clean_prompt}"
            else:
                summary = clean_prompt[:100]
                if len(clean_prompt) > 100:
                    summary += "..."

            step_line = f"{status_sym} Step {i+1}{ts_str} ({action_type}): {summary}"

            repl_id = row.get("repl_id")
            if repl_id:
                step_line += f" (REPL: {repl_id})"

            exec_summary = row.get("execution_summary")
            if exec_summary:
                clean_res = str(exec_summary).strip()
                if len(clean_res) > 200:
                    step_line += (
                        f"\n    -> Result: {clean_res[:200]}... (See rlm.history)"
                    )
                else:
                    step_line += f"\n    -> Result: {clean_res}"

            next_action = row.get("next_action")
            if next_action and str(next_action).strip():
                step_line += f"\n    -> Next: {next_action.strip()}"

            if (
                dreamer_analysis and status != "rejected"
            ):  # Only show if not already summarized in group
                step_line += f"\n    -> Dreamer Analysis: {dreamer_analysis}"

            final_response = row.get("final_response")
            if final_response and str(final_response).strip():
                step_line += f"\n    -> RLM_FINAL_RESPONSE: {final_response.strip()}"

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
                    prompt = (row.get("initial_prompt") or "")[:40]
                    status = row.get("last_status", "unknown")
                    last_activity = row.get("last_activity", "")
                    last_res = (row.get("last_result") or "")[:60]
                    last_act = (row.get("last_action") or "")[:40]
                else:
                    sid = row[0]
                    # count = row[1] # Unused
                    prompt = (row[2] or "")[:40]
                    status = row[4] if len(row) > 4 else "unknown"
                    last_activity = row[3] if len(row) > 3 else ""
                    last_res = (row[5] if len(row) > 5 else "")[:60]
                    last_act = (row[6] if len(row) > 6 else "")[:40]

                if last_res:
                    last_res = f" -> {last_res}..."

                if last_act and last_act != prompt:
                    prompt = f"{prompt}.. > {last_act}"

                # Format timestamp
                ts_str = ""
                if last_activity:
                    try:
                        if isinstance(last_activity, (int, float)):
                            ts = datetime.fromtimestamp(last_activity / 1000)
                            ts_str = ts.strftime("%H:%M:%S")
                    except Exception as e:
                        logger.warning(f"Failed to format sub-REPL timestamp: {e}")

                status_sym = (
                    "🟢"
                    if status == "success"
                    else "🔵" if status == "running" else "🔴"
                )
                lines.append(
                    f"- {status_sym} `{sid[:8]}` | {prompt}{last_res} | {ts_str}"
                )

            return lines

        except Exception as e:
            logger.error(f"Failed to get sub-REPLs: {e}")
            return []

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

                    short_p = (prompt or "")[:60]
                    short_r = (res or "")[:100]
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

        except Exception as e:
            logger.error(f"Failed to build logical audit: {e}")
            return ""


# Singleton instance
scratchpad_builder = ScratchpadBuilder()
