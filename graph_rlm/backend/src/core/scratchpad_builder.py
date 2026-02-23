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
import redis

from .circuit import generate_correlation_id, get_correlation_id
from .config import settings
from .db import GraphClient, db
from .logger import get_logger
from .services.circuit import CircuitOpenError, protected_llm_generate

logger = get_logger("graph_rlm.scratchpad_builder")


# RTM (Random Tree Model) Parameters
# From: "Random Tree Model of Meaningful Memory" (2025)
# K = branching factor, D = max depth
RTM_K = 4
RTM_D = 4
RTM_LEAF_WINDOW = RTM_K ** (RTM_D - 2)  # 16 — recent steps shown individually


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
        execution_state: Optional[Any] = None,
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

        # === 0. MISSION CONTROL (Phase C2) — Agent Orientation ===
        # This structured header tells the stateless agent where it is,
        # what's working, and what to focus on next.
        if execution_state is not None:
            phase = getattr(execution_state, "phase", "EXPLORING")
            consec_ok = getattr(execution_state, "consecutive_successes", 0)
            consec_fail = getattr(execution_state, "consecutive_failures", 0)
            interventions = getattr(execution_state, "intervention_count", 0)
            sheaf_e = getattr(execution_state, "last_sheaf_energy", 0.0)
            omcd_q = getattr(
                execution_state, "last_omcd_qstop", 0.0
            )  # noqa: F841 (used in mc_lines below)
            critique = getattr(execution_state, "last_dreamer_critique", None)
            outcomes = getattr(execution_state, "step_outcomes", [])

            # Build momentum indicator from last 10 outcomes
            momentum_symbols = {
                "success": "✓",
                "completed": "✓",
                "failed": "✗",
                "error": "✗",
                "reflexion": "🔄",
                "running": "⏳",
            }
            momentum_str = (
                "".join(momentum_symbols.get(o, "·") for o in outcomes[-10:])
                or "(no steps yet)"
            )

            # Health assessment
            if consec_fail >= 3:
                health = "⚠️ STRUGGLING (3+ consecutive failures)"
            elif sheaf_e > 0.5:
                health = f"⚠️ DRIFTING (Sheaf Energy: {sheaf_e:.2f})"
            elif consec_ok >= 3:
                health = "✅ STRONG (3+ consecutive successes)"
            else:
                health = "🔵 NOMINAL"

            mc_lines = [
                "## 🎯 Mission Control",
                f"- **Phase**: {phase}",
                f"- **Health**: {health}",
                f"- **Momentum**: {momentum_str} (last {min(len(outcomes), 10)} steps)",
                f"- **Successes/Failures**: {consec_ok} consecutive ✓ / {consec_fail} consecutive ✗",
                f"- **Monitors**: Sheaf Energy={sheaf_e:.2f} | oMCD q_stop={omcd_q:.2f}",
            ]
            if interventions > 0:
                mc_lines.append(
                    f"- **Interventions**: {interventions} (Reflexion/HOT SEAT)"
                )
            if critique:
                mc_lines.append(f"- **Last Dreamer Critique**: {critique[:200]}")
            mc_lines.append("")
            lines.extend(mc_lines)

        # === 1. Previous Rounds (Variable Length Summaries) — STATIC ===
        if completed_rounds:
            lines.append("## Previous Rounds (Compressed Context)")

            # Prepare summaries (Parallel execution for missing ones)
            final_summaries = []
            generation_tasks = []
            task_metadata = []  # (index_in_rounds, field_name)

            for i, r in enumerate(completed_rounds):
                p_sum = r.get("prompt_summary")
                r_sum = r.get("result_summary")

                # We need two slots per round in the final re-assembled list
                if p_sum:
                    p_final = p_sum
                else:
                    p_final = None  # Placeholder
                    generation_tasks.append(
                        self._summarize_content(
                            r.get("user_prompt") or "", "User Prompt"
                        )
                    )
                    task_metadata.append((i, "prompt_summary"))

                # Handle result summary separately to ensure both are checked
                if r_sum:
                    r_final = r_sum
                else:
                    r_final = None  # Placeholder
                    generation_tasks.append(
                        self._summarize_content(
                            r.get("final_response") or "", "Agent Result"
                        )
                    )
                    task_metadata.append((i, "result_summary"))

                final_summaries.append({"prompt": p_final, "result": r_final})

            # Execute missing summaries
            if generation_tasks:
                generated_vals = await asyncio.gather(*generation_tasks)

                # Map back and persist
                for val, (round_idx, field) in zip(
                    generated_vals, task_metadata, strict=True
                ):
                    if field == "prompt_summary":
                        final_summaries[round_idx]["prompt"] = val
                    else:
                        final_summaries[round_idx]["result"] = val

                    # Persist if BOTH are now present (to avoid multiple DB calls, or just do it individually)
                    current_r = completed_rounds[round_idx]
                    rid = current_r.get("round_id")
                    if rid:
                        p_val = final_summaries[round_idx]["prompt"]
                        r_val = final_summaries[round_idx]["result"]
                        if p_val and r_val:
                            try:
                                self.db.update_round_summaries(rid, p_val, r_val)
                            except Exception as db_e:
                                logger.warning(
                                    "Failed to persist round summaries for %s: %s",
                                    rid,
                                    db_e,
                                )

            # Re-assemble for display
            for i, r in enumerate(completed_rounds, 1):
                rid = r.get("round_id") or "unknown"
                repl_ids = r.get("repl_ids") or []
                repls_str = ", ".join(repl_ids) if repl_ids else "None"

                prompt_summary = final_summaries[i - 1]["prompt"] or "(summary failed)"
                result_summary = final_summaries[i - 1]["result"] or "(summary failed)"

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
                f"INPUT: \n---\n{text[:100000]}\n---\n\n"
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
                match = re.search(r"```(?:json)?\n(.*?)\n```", json_str, re.DOTALL)
                if match:
                    json_str = match.group(1).strip()
                elif json_str.find("[") != -1 and json_str.rfind("]") != -1:
                    json_str = json_str[json_str.find("[") : json_str.rfind("]") + 1]

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

        # --- RTM HIERARCHICAL COMPRESSION ---
        # Split into gist groups (old, compressed) + leaf rows (recent, detailed)
        gist_chunks, leaf_rows = self._build_rtm_tree(processed_data)

        # Generate summaries: gist chunks get ONE summary per group,
        # leaf rows get individual summaries (same as before)
        gist_tasks = [self._generate_gist_summary(chunk) for chunk in gist_chunks]
        leaf_tasks = [self._generate_step_summary(row) for row in leaf_rows]

        gist_summaries = await asyncio.gather(*gist_tasks) if gist_tasks else []
        leaf_summaries = await asyncio.gather(*leaf_tasks)

        # Merge into unified view: gist rows first, then leaf rows
        # windowed_data and summaries are rebuilt from the RTM tree
        windowed_data = []
        summaries = []

        for i, chunk in enumerate(gist_chunks):
            # Create a synthetic "gist row" from chunk metadata
            first = chunk[0]
            last = chunk[-1]
            step_start = first.get("step_id", "?")
            step_end = last.get("step_id", "?")
            turn = first.get("turn_id", "?")

            # Average sheaf/repe scores across chunk
            def _avg(key, _chunk=chunk):
                vals = [
                    float(r.get(key, 0) or 0) for r in _chunk if r.get(key) is not None
                ]
                return round(sum(vals) / len(vals), 3) if vals else None

            gist_row = {
                "id": f"gist:{first.get('id', '?')[:8]}..{last.get('id', '?')[:8]}",
                "created_at": first.get("created_at"),
                "repl_id": first.get("repl_id"),
                "turn_id": turn,
                "step_id": f"{step_start}-{step_end}",
                "status": "gist",
                "sheaf_score": _avg("sheaf_score"),
                "repe_shakiness": _avg("repe_shakiness"),
                "repe_evasion": _avg("repe_evasion"),
                "repe_confluence": _avg("repe_confluence"),
                "repe_freedom": _avg("repe_freedom"),
                "omcd_score": _avg("omcd_score"),
                "h0_rank": _avg("h0_rank"),
                "_is_gist": True,
                "_gist_count": len(chunk),
            }
            windowed_data.append(gist_row)
            summaries.append(gist_summaries[i])

        # Append leaf rows as-is
        for i, row in enumerate(leaf_rows):
            windowed_data.append(row)
            summaries.append(leaf_summaries[i])

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
                "gist": "📦",
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
            def safe_float(v):
                if v is None:
                    return None
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return None

            shaky = safe_float(row.get("repe_shakiness"))
            evasion = safe_float(row.get("repe_evasion"))
            confluence = safe_float(row.get("repe_confluence"))
            freedom = safe_float(row.get("repe_freedom"))
            sheaf = safe_float(row.get("sheaf_score"))
            omcd = safe_float(row.get("omcd_score"))
            h0_rank = safe_float(row.get("h0_rank"))

            # Format scores safely with condensed labels
            def fmt(v):
                return f"{v:.2f}" if v is not None else "--"

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
            if isinstance(h0_rank, (int, float)) and h0_rank > 0.7:
                alerts.append("!! H0_RANK HIGH !!")

            ratings = gestalt
            if alerts:
                ratings = f"**{gestalt}**<br/>" + " ".join(alerts)

            # Recall
            tid = row.get("id")
            recall_link = f"`recall('{tid}')`"

            # Code Hash (for introspection)
            chash = (row.get("code_hash") or "       ")[:7]

            lines.append(
                f"| {ts_str} | `{repl}` | {ts_display} | {st} | `{chash}` | {ratings} | {summary} | {recall_link} |"
            )

        return "\n".join(lines)

    def _build_rtm_tree(self, results: List[Dict]) -> tuple:
        """
        RTM Hierarchical Compression with Gain Modulation.

        Inspired by Li et al. (2025) "Neural Mechanisms of Resource Allocation
        in Working Memory": priority-based gain modulation determines which
        items get high-fidelity representation (leaves) vs compressed (gist).

        Gain = recency_weight × consistency_weight
        - Recency: newer steps get higher weight (positional)
        - Consistency: low sheaf_score = high topological consistency = higher gain

        High-gain steps become leaves (detailed), low-gain become gist (compressed).
        Max leaf count capped at RTM_K^(RTM_D-1) = 64.

        Returns:
            (gist_chunks, leaf_rows) where gist_chunks is List[List[Dict]]
            and leaf_rows is List[Dict]
        """
        total = len(results)
        max_leaves = RTM_K ** (RTM_D - 1)  # 64

        if total <= RTM_LEAF_WINDOW:
            # Short session — everything is a leaf (full detail)
            return [], results

        # --- GAIN MODULATION ---
        # Compute gain score for each step: recency × consistency
        gains = []
        for i, row in enumerate(results):
            # Recency: linear ramp from 0.1 (oldest) to 1.0 (newest)
            recency = 0.1 + 0.9 * (i / max(total - 1, 1))

            # Consistency: invert sheaf_score (low sheaf = high consistency = high gain)
            sheaf = None
            raw = row.get("sheaf_score")
            if raw is not None:
                try:
                    sheaf = float(raw)
                except (ValueError, TypeError):
                    sheaf = None

            if sheaf is not None:
                consistency = 1.0 - min(sheaf, 1.0)  # 0→1, 1→0
            else:
                consistency = 0.5  # Unknown sheaf = neutral

            gain = recency * consistency
            gains.append(gain)

        # Determine adaptive leaf window:
        # Start with default RTM_LEAF_WINDOW, then adjust based on gain distribution
        # Steps with gain above the median are candidates for leaf status
        sorted_gains = sorted(gains)
        median_gain = sorted_gains[len(sorted_gains) // 2] if sorted_gains else 0.5

        # Count how many steps exceed median gain (these want to be leaves)
        above_median = sum(1 for g in gains if g >= median_gain)
        # Adaptive window: at least RTM_LEAF_WINDOW, up to max_leaves,
        # biased by how many high-gain steps exist
        adaptive_window = max(RTM_LEAF_WINDOW, min(above_median, max_leaves))

        # But we still need at least SOME gist (don't make everything a leaf)
        if adaptive_window >= total:
            return [], results

        # Select the top adaptive_window steps BY GAIN, but preserve temporal order
        # Pair each step with (gain, original_index)
        indexed_gains = [(gains[i], i) for i in range(total)]
        # Sort by gain descending, take top adaptive_window
        top_indices = sorted(
            [
                idx
                for _, idx in sorted(indexed_gains, key=lambda x: -x[0])[
                    :adaptive_window
                ]
            ]
        )

        # Build leaf and gist sets, preserving temporal order
        leaf_set = set(top_indices)
        gist_rows = []
        leaf_rows = []

        for i, row in enumerate(results):
            if i in leaf_set:
                leaf_rows.append(row)
            else:
                gist_rows.append(row)

        # Group gist rows into chunks of K
        chunks = []
        for i in range(0, len(gist_rows), RTM_K):
            chunk = gist_rows[i : i + RTM_K]
            chunks.append(chunk)

        # If there are still too many chunks, recursively compress
        if len(chunks) > max_leaves:
            super_cutoff = len(chunks) - max_leaves
            super_old = chunks[:super_cutoff]
            remaining = chunks[super_cutoff:]

            merged = []
            for i in range(0, len(super_old), RTM_K):
                mega = []
                for sub in super_old[i : i + RTM_K]:
                    mega.extend(sub)
                merged.append(mega)

            chunks = merged + remaining

        return chunks, leaf_rows

    async def _generate_gist_summary(self, chunk: List[Dict]) -> str:
        """
        Generate a single compressed summary for a group of RTM steps.

        This is the RTM "interior node" — a gist that compresses K steps
        into one summary, preserving key outcomes and actions.
        """
        # Build a condensed representation of the chunk
        step_briefs = []
        for row in chunk:
            prompt = (row.get("prompt") or "")[:10000]
            result = (row.get("result") or "")[:10000]
            status = row.get("status", "?")

            # Use execution_summary if available (cheaper)
            if row.get("execution_summary"):
                brief = str(row["execution_summary"])[:500]
            elif prompt:
                brief = f"[{status}] {prompt[:5000]}"
                if result:
                    brief += f" → {result[:5000]}"
            else:
                brief = f"[{status}] (no prompt)"
            step_briefs.append(brief)

        combined = "\n".join(step_briefs)
        count = len(chunk)

        try:
            summary_model = settings.SUMMARY_MODEL or "gemini-2.0-flash-lite"
            llm_prompt = f"""Compress these {count} agent steps into ONE dense summary (max 200 chars).
Focus on: what was DONE, what WORKED, what FAILED. Be specific with file names and values.
Do NOT use filler phrases. This is a memory compression for the agent's scratchpad.

STEPS:
{combined[:30000]}

Compressed summary:"""

            summary = await protected_llm_generate(
                llm_prompt,
                model=summary_model,
                correlation_id=get_correlation_id() or generate_correlation_id(),
            )
            return f"[{count} steps] {summary.strip()[:150].replace(chr(10), ' ')}"
        except (httpx.RequestError, ValueError, RuntimeError):
            # Fallback: just list statuses
            statuses = [r.get("status", "?") for r in chunk]
            return f"[{count} steps] {', '.join(statuses)}"

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

        # Use cached step_summary if available from DB
        cached_summary = row.get("step_summary")
        if cached_summary:
            return str(cached_summary).strip()[:120].replace("\n", " ")

        # Use execution_summary if available from DB (cheaper second-best)
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
ACTION: {prompt[:50000]}
RESULT: {result[:50000]}
STATUS: {status}

Summary:"""

            summary = await protected_llm_generate(
                llm_prompt,
                model=summary_model,
                correlation_id=get_correlation_id() or generate_correlation_id(),
            )
            final_summary = summary.strip()[:120].replace("\n", " ")

            # Persist the newly generated summary to the DB
            thought_id = row.get("id")
            if thought_id:
                try:
                    self.db.update_thought_result(
                        thought_id=thought_id,
                        result=result,
                        step_summary=final_summary,
                        status=status,
                    )
                except (RuntimeError, ValueError, redis.exceptions.RedisError) as db_e:
                    logger.warning(
                        "Failed to persist step summary for %s: %s", thought_id, db_e
                    )

            return final_summary
        except (CircuitOpenError, httpx.RequestError, ValueError, RuntimeError) as e:
            logger.debug("Summarization failed: %s", e)
            # Fallback
            return prompt.strip()[:100].replace("\n", " ")
        except (AttributeError, TypeError) as e:
            logger.warning("Unexpected error in _generate_step_summary: %s", e)
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

    def _build_missing_requirements(self, _task: str, gestalt: str) -> str:
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
