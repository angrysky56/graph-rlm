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
import json
import re
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

        Uses round-based architecture optimized for prefix caching:
        - Static/Append-only sections at the top.
        - Dynamic session state / grounding at the bottom.
        """
        lines = []

        # Count completed rounds
        completed_rounds = self.db.get_completed_rounds(root_session_id)
        current_round_num = len(completed_rounds) + 1

        # === 1. Previous Rounds (Variable Length Summaries) — STATIC ===
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

        # === 2. Current Round Progress (The Trace) — APPEND-ONLY ===
        progress = await self._build_current_round_progress(
            session_id, root_session_id, current_round_id
        )
        lines.append("## Execution Trace (Current Round)")
        lines.append(progress)
        lines.append("")

        # === 3. Thimac Memory Gestalt (Grounding Anchor) — SEMI-DYNAMIC ===
        if morph_gestalt:
            lines.append("## 🧠 THIMAC GESTALT (State Anchor)")
            lines.append("> [!] Ontological State of the Session Memory")
            lines.append(morph_gestalt)
            lines.append("-" * 40)
            lines.append("")

        # === 4. Missing Requirements (Cognitive Gap) — DYNAMIC ===
        if morph_gestalt:
            missing = self._build_missing_requirements(task, morph_gestalt)
            if missing:
                lines.append("## Missing Requirements (The Gap)")
                lines.append(missing)
                lines.append("")

        # === 7. Grounding Header (Highly Dynamic) — MOVED TO BOTTOM ===
        now = datetime.now(timezone.utc)
        header_lines = []
        header_lines.append("## Agent Session State (System TUI)")
        header_lines.append(
            f"- **Time**: {now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} UTC"
        )

        state_header = f"Session: {session_id[:8]}... | Round: {current_round_num} | Step: {current_step}/{max_steps}"
        if current_repl_id:
            state_header += f" | Active REPL: {current_repl_id}"

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
            state_header += f" | Last Successful REPL: {last_repl}"

        header_lines.append(state_header)
        header_lines.append("=" * 60)

        lines.extend(header_lines)

        return "\n".join(lines)

    async def _summarize_content(self, text: str, label: str) -> str:
        """
        Summarize content using the LLM with the "Chain of Density" technique.
        Generates increasingly entity-dense summaries and returns the final iteration.
        """
        if not text:
            return "(empty)"

        # Threshold: Don't waste LLM calls on short text
        if len(text) < 2000:
            return text.strip()

        try:
            summary_model = settings.SUMMARY_MODEL or "gemini-2.0-flash-lite"

            prompt = (
                f"You will generate increasingly concise, entity-dense summaries of the following {label}.\n\n"
                f"INPUT: \n---\n{text[:8000]}\n---\n\n"
                "Repeat the following 2 steps 4 times.\n"
                "Step 1. Identify 1-3 informative Entities ('; ' delimited) from the Input which are missing from the previously generated summary.\n"
                "Step 2. Write a new, denser summary that covers every entity and detail from the previous summary plus the new Missing Entities.\n\n"
                "Guidelines:\n"
                "- The first summary should be comprehensive but broadly written (4-5 sentences, roughly 70-90 words).\n"
                "- Make every word count: iteratively rewrite to improve flow and make space for additional entities.\n"
                "- Compress the text by fusing ideas and removing redundant phrases.\n"
                "- READABILITY: The summaries must remain highly readable and grammatically coherent.\n"
                "- LENGTH: Keep the length strictly between 70 and 90 words for every summary.\n\n"
                'Answer in JSON. The JSON should be a list (length 4) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary".'
            )

            response = await protected_llm_generate(
                prompt,
                model=summary_model,
                correlation_id=get_correlation_id() or generate_correlation_id(),
            )

            # Extract the densest summary from the JSON response
            try:
                # Handle potential markdown code blocks in the response
                json_str = response.strip()
                if json_str.startswith("```"):
                    # Use regex to find the content inside triple backticks
                    match = re.search(r"```(?:json)?\n(.*?)\n```", json_str, re.DOTALL)
                    if match:
                        json_str = match.group(1)

                data = json.loads(json_str)
                if isinstance(data, list) and len(data) > 0:
                    # Return the last (densest) summary
                    final_summary = data[-1].get("Denser_Summary", "")
                    if final_summary:
                        return final_summary.strip()

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as je:
                logger.debug(
                    "JSON parsing of CoD summary failed: %s. Falling back to raw response.",
                    je,
                )
                # Fallback: if it's not JSON but looks like a summary, use it
                if len(response) > 50 and "{" not in response:
                    return response.strip()

            return response.strip()

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
                   n.code_hash as code_hash,
                   n.sheaf_score as sheaf_score,
                   n.h0_rank as h0_rank,
                   n.repe_shakiness as repe_shakiness,
                   n.repe_evasion as repe_evasion,
                   n.repe_confluence as repe_confluence,
                   n.repe_freedom as repe_freedom,
                   n.omcd_score as omcd_score,
                   n.thimac_op as thimac_op,
                   n.thimac_level as thimac_level,
                   n.navigator_insight as navigator_insight
            ORDER BY n.created_at ASC
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
                }
                # Check bounds for new columns
                if len(row) > 8:
                    base_data["code_hash"] = row[8]
                if len(row) > 9:
                    base_data["sheaf_score"] = row[9]
                if len(row) > 10:
                    base_data["h0_rank"] = row[10]
                if len(row) > 11:
                    base_data["repe_shakiness"] = row[11]
                if len(row) > 12:
                    base_data["repe_evasion"] = row[12]
                if len(row) > 13:
                    base_data["repe_confluence"] = row[13]
                if len(row) > 14:
                    base_data["repe_freedom"] = row[14]
                if len(row) > 15:
                    base_data["omcd_score"] = row[15]
                if len(row) > 16:
                    base_data["thimac_op"] = row[16]
                if len(row) > 17:
                    base_data["thimac_level"] = row[17]
                if len(row) > 18:
                    base_data["navigator_insight"] = row[18]

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
            "| Time | REPL | T.S | St | Gestalt (Ψ,📐,Ω,🧭) | Summary (Action & Outcome) | Recall ID |"
        )
        lines.append("|---|---|---|---|---|---|---|")

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

            # --- LOGICAL KNOT AUDIT (Triplets) ---
            if row.get("status") in ["failed", "error"]:
                triplet = self._extract_failure_triplet(row)
                if triplet:
                    summary = f"**{summary}**<br/>`{triplet}`"

            # --- OBSERVABILITY RATINGS (Ψ,📐,Ω,🧭) ---
            shaky = row.get("repe_shakiness")
            evasion = row.get("repe_evasion")
            confluence = row.get("repe_confluence")
            freedom = row.get("repe_freedom")
            sheaf = row.get("sheaf_score")
            omcd = row.get("omcd_score")
            h0_rank = row.get("h0_rank")

            # Format scores safely with condensed labels
            def fmt(v):
                return f"{v:.2f}" if isinstance(v, (float, int)) else "--"

            shaky_str = fmt(shaky)
            confluence_str = fmt(confluence)
            evasion_str = fmt(evasion)
            freedom_str = fmt(freedom)
            sheaf_str = fmt(sheaf)
            omcd_str = fmt(omcd)
            h0_rank_str = fmt(h0_rank)

            # Gestalt string with RepE axes + Sheaf Consistency + oMCD Stop Probability
            # S=Shakiness, C=Confluence, E=Evasion, F=Freedom
            gestalt = f"Ψ(S:{shaky_str} C:{confluence_str} E:{evasion_str} F:{freedom_str}) | 📐(S:{sheaf_str} H0:{h0_rank_str}) | Ω:{omcd_str}"

            nav_insight = row.get("navigator_insight")
            if nav_insight:
                gestalt += f" | 🧭 {nav_insight}"

            # --- ALERT DECORATION ---
            alerts = []
            if isinstance(sheaf, (int, float)) and sheaf > 0.7:
                alerts.append("!! LOOP !!")
            if isinstance(evasion, (int, float)) and evasion < -0.15:
                alerts.append("!! EVASION !!")
            if isinstance(shaky, (int, float)) and shaky < -0.15:
                alerts.append("!! UNCERTAIN !!")
            if isinstance(omcd, (int, float)) and omcd < 0.3:
                alerts.append("!! LOW CONFIG !!")
            if isinstance(h0_rank, (int, float)) and h0_rank > 0.7:
                alerts.append("!! H0_RANK HIGH !!")

            ratings = gestalt
            if alerts:
                ratings = f"**{gestalt}**<br/>" + " ".join(alerts)

            # Recall
            tid = row.get("id")
            recall_link = f"`recall('{tid}')`"

            lines.append(
                f"| {ts_str} | `{repl}` | {ts_display} | {st} | {ratings} | {summary} | {recall_link} |"
            )

        return "\n".join(lines)

    async def _generate_step_summary(self, row: Dict) -> str:
        """
        Use SUMMARY_MODEL to generate an entity-dense summary of the step.
        """
        prompt = row.get("prompt") or ""
        status = row.get("status")

        if status in [
            "navigator",
            "omcd",
            "sheaf",
            "repe",
            "reflexion",
            "system",
            "fragment",
        ]:
            return prompt.strip()[:100]

        # Use execution_summary if available from DB (cheaper)
        if row.get("execution_summary"):
            summary = str(row.get("execution_summary"))
            if len(summary) > 80 and "/" in summary:
                parts = summary.split("/")
                if len(parts) > 3:
                    # Keep the root and the filename, compress the middle
                    summary = f"{parts[0]}/.../{parts[-2]}/{parts[-1]}"
            return summary[:120].replace("\n", " ")

        prompt = row.get("prompt") or ""
        result = row.get("result") or ""
        status = row.get("status")

        # Fast path for very short items
        if len(prompt) < 60 and len(result) < 60:
            return prompt.strip()[:80]

        try:
            summary_model = settings.SUMMARY_MODEL or "gemini-2.0-flash-lite"

            llm_prompt = f"""Summarize this agent step professionally (max 100 chars).
Focus on specific names, values, and the concrete outcome.
BE DENSE: Avoid fillers like 'The agent...'.
ACTION: {prompt[:1000]}
RESULT: {result[:1000]}
STATUS: {status}

Summary:"""

            summary = await protected_llm_generate(
                llm_prompt,
                model=summary_model,
                correlation_id=get_correlation_id() or generate_correlation_id(),
            )
            return summary.strip()[:120].replace("\n", " ")
        except (httpx.RequestError, ValueError, RuntimeError):
            # Fallback
            return prompt.strip()[:100].replace("\n", " ")

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

    def _extract_failure_triplet(self, row: Dict) -> str:
        """
        Extract a (Subject, Relation, Object) triplet from a failing thought.
        """
        try:
            result = (row.get("result") or "").lower()

            subject = "Agent"
            relation = "Failure"
            target = "Goal"

            if "error" in result:
                relation = result.split(":")[0].strip() if ":" in result else "Error"

                # Try to extract target from quoted strings first (usually the object/file)

                quotes = re.findall(r"['\"](.*?)['\"]", result)
                if quotes:
                    target = quotes[0]
                else:
                    # Fallback to last word
                    words = result.split()
                    if len(words) > 2:
                        target = words[-1].strip("'\".,")

            return f"{subject} -> {relation} -> {target}"
        except (AttributeError, IndexError, ValueError):
            return ""

    def _build_missing_requirements(self, task: str, gestalt: str) -> str:
        """
        Detects missing requirements based on empirical session issues.
        Replaces guessing words from prompt with actual system-detected gaps.
        """
        issues = []

        # 1. Existence Gap (The primary "Missing Requirement")
        if "EXISTENCE: No materialized results" in gestalt:
            issues.append(
                "Materialized Existence: No concrete files or artifacts generated yet."
            )

        # 2. Sheaf Fragment (Logical Knots)
        # We check the gestalt for h0_rank alerts or energy spikes if we had them here,
        # but better to query recent history for high h0_rank.
        if "CHAOTIC" in gestalt:
            issues.append(
                "Logical Consistency: Topological stress detected (Chaotic state)."
            )

        # 3. Navigator Stalls
        if "Progress: 0.0000" in gestalt:
            issues.append(
                "Intrinsic Progress: Thinking is stagnant or repeating (Low Compression Progress)."
            )

        # 4. oMCD Low Confidence
        # (Could extract from Ω value if we parsed it back)

        if issues:
            return "\n".join([f"> [!] Missing: {iss}" for iss in issues])

        return "> [✓] Operational Alignment: All systems reporting normal grounding and progress."


# Singleton instance
scratchpad_builder = ScratchpadBuilder()
