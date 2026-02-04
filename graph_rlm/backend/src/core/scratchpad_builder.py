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

from .database import client
from .logger import get_logger

logger = get_logger("graph_rlm.scratchpad_builder")


class ScratchpadBuilder:
    """Builds a structured scratchpad for the stateless agent."""

    def __init__(self):
        pass

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
        """
        lines = []

        # === Header with current datetime ===
        now = datetime.now(timezone.utc)
        local_now = datetime.now()

        # Count completed rounds for round number display
        completed_rounds = client.repo.get_completed_rounds(root_session_id)
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
        recent_nodes = client.repo.get_context_frontier(session_id, limit=3)
        if recent_nodes:
            lines.append("## Immediate Graph Context")
            for node in recent_nodes:
                summary = (node.get("execution_summary") or "No summary")[:100]
                status = node.get("status")
                nid = node.get("id")
                lines.append(f"- [{nid}] ({status}): {summary}")
            lines.append("")

        # === Current Task ===
        lines.append("## Current Task")
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
        """
        try:
            results = client.repo.get_current_round_thoughts(root_session_id, current_round_id)

            if not results:
                return "No progress recorded yet."

            return self._format_progress_rows(results)

        except Exception as e:
            logger.error(f"Failed to build current round progress: {e}")
            return f"Error loading progress: {e}"

    def _format_progress_rows(self, results: List[Any]) -> str:
        """
        Helper to format detailed progress rows from db results.
        """
        lines = []
        # Repo returns List[Dict] already
        processed_data = results

        # 2. Group and Format
        i = 0
        while i < len(processed_data):
            row = processed_data[i]
            status = row.get("status") or "unknown"
            dreamer_analysis = (row.get("dreamer_analysis") or "").strip()

            # Condensation Logic
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
                summary = prompt_lines[0][:80] if prompt_lines else ""
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
            ):
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
            results = client.repo.get_sub_repls_data(root_session_id, current_session_id)

            lines = []
            for row in results:
                sid = row.get("sid", "unknown")
                prompt = (row.get("initial_prompt") or "")[:40]
                status = row.get("last_status", "unknown")
                last_activity = row.get("last_activity", "")
                last_res = (row.get("last_result") or "")[:60]
                last_act = (row.get("last_action") or "")[:40]

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
        """
        try:
            lines = []

            # 1. Active Knots
            active_res = client.repo.get_active_failure_knots(root_session_id, session_id)

            if active_res:
                lines.append("### 🔴 Active Failure Knots (Needs Fix)")
                for row in active_res:
                    rid = row.get("id")
                    prompt = row.get("prompt")
                    res = row.get("result")

                    short_p = (prompt or "")[:60]
                    short_r = (res or "")[:100]
                    lines.append(f"- [!] `{rid}`: {short_p} -> {short_r}")
            else:
                lines.append("### 🟢 Logical State Clean (No Active Failures)")

            # 2. Recently Resolved Knots
            resolved_res = client.repo.get_resolved_failure_knots(root_session_id, session_id)

            if resolved_res:
                lines.append("")
                lines.append("### ✅ Recently Resolved (Dreamer Acknowledged)")
                for row in resolved_res:
                    prompt = row.get("prompt")
                    lines.append(f"- [x] {(prompt or '')[:60]}...")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Failed to build logical audit: {e}")
            return ""


# Singleton instance
scratchpad_builder = ScratchpadBuilder()
