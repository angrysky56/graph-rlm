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
    ) -> str:
        """
        Build a complete scratchpad for the agent.

        Uses round-based architecture:
        - Previous rounds: Compressed summaries with REPL ID pointers
        - Current round: Full detail of progress (System TUI Table)
        - Thimac Gestalt: Existence/Subsistence overview

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

        # Count completed rounds for round number display
        completed_rounds = self.db.get_completed_rounds(root_session_id)
        current_round_num = len(completed_rounds) + 1

        lines.append("## Agent Session State (System TUI)")
        lines.append(f"- **Time**: {now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} UTC")
        lines.append(f"- **Session**: `{session_id}`")
        lines.append(
            f"- **Round**: {current_round_num} | **Step**: {current_step}/{max_steps}"
        )
        lines.append("")

        # === Previous Rounds (COMPRESSED) ===
        if completed_rounds:
            lines.append("## Previous Rounds (Compressed)")
            lines.append("| # | Round ID | Prompt | REPLs | Result |")
            lines.append("|---|----------|--------|-------|--------|")
            for i, r in enumerate(completed_rounds, 1):
                rid = r.get("round_id") or "unknown"
                prompt = (r.get("user_prompt") or "")[:100]
                repl_ids = r.get("repl_ids") or []
                repls = ", ".join(repl_ids[:2])
                if len(repl_ids) > 2:
                    repls += f" +{len(repl_ids)-2}"
                result = (r.get("final_response") or "")[:100]
                lines.append(
                    f"| {i} | `{rid[:8]}` | {prompt}... | {repls} | {result}... |"
                )
            lines.append("")

        # === Logical State Audit (Failures & Resolutions) ===
        audit_section = self._build_logical_audit(session_id, root_session_id)
        if audit_section:
            lines.append("## Logical State Audit (Knots)")
            lines.append(audit_section)
            lines.append("")

        # === Current Task ===
        lines.append(f"## Current Task: {task}")
        lines.append("")

        # === Current Round Progress (System TUI Table) ===
        progress_section = await self._build_current_round_progress(
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

        # === Thimac Memory Gestalt ===
        # Replaces the old Morphological Memory vector dump
        if morph_gestalt:
            lines.append(morph_gestalt)
            lines.append("")

        # === Recall Instructions ===
        lines.append("## Data Commands")
        lines.append(
            "- `await rlm.recall('ID')`: Retrieve full node content (Code/Result)"
        )
        lines.append("- `await rlm.search('query')`: Semantic search across history")

        return "\n".join(lines)

    async def _build_current_round_progress(
        self, session_id: str, root_session_id: str, _current_round_id: str
    ) -> str:
        """
        Build progress for ONLY the current round (not archived rounds).
        """
        try:
            # Get ALL thoughts for current session
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
                   n.turn_id as turn_id,
                   n.step_id as step_id
            ORDER BY n.turn_id ASC, n.step_id ASC, n.created_at ASC
            """
            results = self.db.query(
                q,
                {"rsid": root_session_id, "sid": session_id},
            )

            if not results:
                return "No progress recorded yet."

            return await self._format_progress_rows(results)

        except (AttributeError, RuntimeError, ValueError) as e:
            logger.error("Failed to build current round progress: %s", e)
            return f"Error loading progress: {e}"

    async def _format_progress_rows(self, results: List[Any]) -> str:
        """
        Format progress as a clean System TUI table.
        Uses SUMMARY_MODEL to generate concise summaries for each step.
        """
        processed_data = []
        for row in results:
            if not row:
                continue
            if isinstance(row, dict):
                processed_data.append(row)
            else:
                processed_data.append(
                    {
                        "id": row[0],
                        "prompt": row[1],
                        "status": row[2],
                        "result": row[3],
                        "created_at": row[4],
                        "repl_id": row[5],
                        "turn_id": row[6],
                        "step_id": row[7],
                    }
                )

        if not processed_data:
            return "No progress rows."

        # Prepare batch LLM tasks for summarization
        # We summary EVERY step to ensure high fidelity
        summary_tasks = []
        for row in processed_data:
            summary_tasks.append(self._generate_step_summary(row))

        # Parallel execution
        summaries = await asyncio.gather(*summary_tasks)

        # Build Table
        lines = []
        lines.append(
            "| Time (ms) | REPL | T.S | St | Summary (Auto-Generated) | Recall ID |"
        )
        lines.append("|---|---|---|---|---|---|")

        for idx, row in enumerate(processed_data):
            # Time
            ts_str = "--:--:--.---"
            if row.get("created_at"):
                dt = datetime.fromtimestamp(row["created_at"] / 1000)
                ts_str = dt.strftime("%H:%M:%S.%f")[:-3]

            # REPL
            repl = (row.get("repl_id") or "       ")[:7]

            # T.S
            turn = row.get("turn_id")
            step = row.get("step_id")
            ts_display = f"{turn}.{step}" if turn is not None else "??"

            # Status
            st_map = {"success": "✓", "failed": "✗", "running": "⏳", "pending": "⋯"}
            st = st_map.get(row.get("status"), "?")

            # Summary
            summary = summaries[idx]

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
            return summary.strip()[:60].replace("\n", " ")
        except (httpx.RequestError, ValueError, RuntimeError):
            # Fallback
            return prompt.strip()[:60].replace("\n", " ")

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
                lines.append("### 🔴 Active Failure Knots")
                for row in active_res:
                    if isinstance(row, dict):
                        rid = row.get("n.id") or row.get("id")
                        prompt = row.get("n.prompt") or row.get("prompt")
                    else:
                        rid = row[0]
                        prompt = row[1]

                    lines.append(f"- `recall('{rid}')`: {(prompt or '')[:60]}...")

            return "\n".join(lines)
        except (AttributeError, RuntimeError, ValueError):
            return ""


# Singleton instance
scratchpad_builder = ScratchpadBuilder()
