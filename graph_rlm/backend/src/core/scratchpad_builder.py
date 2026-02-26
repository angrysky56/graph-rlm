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

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .db import GraphClient, db
from .logger import get_logger
from .semantic_summarizer import summarize_event

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
        memory_trajectory: Optional[List[Any]] = None,
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

            # --- PHASE C3: Rationales ---
            sheaf_rat = getattr(execution_state, "last_sheaf_rationale", None)
            omcd_rat = getattr(execution_state, "last_omcd_rationale", None)
            if sheaf_rat:
                mc_lines.append(f"  - 📐 **Sheaf Rationale**: {sheaf_rat}")
            if omcd_rat:
                mc_lines.append(f"  - Ω **oMCD Rationale**: {omcd_rat}")

            if interventions > 0:
                mc_lines.append(
                    f"- **Interventions**: {interventions} (Reflexion/HOT SEAT)"
                )
            lines.extend(mc_lines)

        # Always append HUD Scale Legend for grounding
        lines.append("")
        lines.append("### 📊 HUD Scale Legend")
        lines.append(
            "- **Gestalt Metrics**: Ψ(S:Shaky, C:Confluent, E:Evasive, F:Free)"
        )
        lines.append(
            "- **Consistency (📐)**: Energy (0.0=Stable, 1.0=Chaotic) | H0 (1=Unified, >1=Fragmented)"
        )
        lines.append("- **Termination (Ω)**: Stop Prob (0.0=Continue, 1.0=Converged)")
        lines.append("")

        # === 1. Previous Rounds (Variable Length Summaries) — STATIC ===
        if completed_rounds:
            lines.append("## Previous Rounds (Compressed Context)")

            # Re-assemble for display
            for i, r in enumerate(completed_rounds, 1):
                rid = r.get("round_id") or "unknown"
                repl_ids = r.get("repl_ids") or []
                repls_str = ", ".join(repl_ids) if repl_ids else "None"

                lines.append(f"### Round {i} (ID: `{rid}`)")
                lines.append("**User Prompt (Raw)**:")
                lines.append(r.get("user_prompt") or "(empty)")
                lines.append("")
                lines.append("**Agent Result (Raw)**:")
                lines.append(r.get("final_response") or "(empty)")
                lines.append("")
                lines.append(f"**REPLs**: `{repls_str}`")
                lines.append("")

        # === 2. Current Round Progress (The Trace) — APPEND-ONLY ===
        progress = await self._build_current_round_progress(
            session_id, root_session_id, current_round_id, memory_trajectory
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

    async def _summarize_content(self, text: str) -> str:
        """Structural summarization using SemanticSummarizer."""
        try:
            return await summarize_event("", text)
        except Exception:  # pylint: disable=broad-except # noqa: BLE001
            logger.warning("SemanticSummarizer failed, falling back to truncation", exc_info=True)
            return text[:500] + "..." if len(text) > 500 else text

    async def _build_current_round_progress(
        self,
        session_id: str,
        root_session_id: str,
        current_round_id: str,
        memory_trajectory: Optional[List[Any]] = None,
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
                   n.navigator_insight as navigator_insight,
                   n.semantic_gist as semantic_gist
            ORDER BY n.created_at ASC
            LIMIT 2000
            """
            results = self.db.query(
                q,
                {"rsid": root_session_id, "sid": session_id, "rid": current_round_id},
            )

            # --- MERGE IN-MEMORY TRAJECTORY ---
            # This is critical because of the Batch Consolidation Flush.
            # Thoughts remain in RAM until ACCEPT/RELEASE.
            if memory_trajectory:
                # Convert memory objects to result dicts for the formatter
                mem_dicts = []
                for ev in memory_trajectory:
                    # Filter for only relevant round/session if trajectory contains more
                    if (
                        str(ev.session_id) != session_id
                        or str(ev.round_id) != current_round_id
                    ):
                        continue

                    d = ev.to_dict()
                    # Ensure alignment with DB field names
                    d["created_at"] = ev.timestamp
                    d["prompt"] = ev.full_data
                    d["thimac_op"] = (
                        ev.operation.value
                        if hasattr(ev.operation, "value")
                        else str(ev.operation)
                    )
                    d["thimac_level"] = (
                        ev.level.value if hasattr(ev.level, "value") else str(ev.level)
                    )
                    mem_dicts.append(d)

                if not results:
                    results = mem_dicts
                else:
                    # Merge and deduplicate by ID, prioritizing memory (latest status)
                    merged = {r["id"]: r for r in results}
                    for md in mem_dicts:
                        merged[md["id"]] = md
                    results = sorted(
                        merged.values(), key=lambda x: x.get("created_at") or 0
                    )

            if not results:
                # FALLBACK: If round-specific query fails, get most recent turns for this session
                # This prevents "amnesia" during strategy shifts or round-id mismatches.
                q_fallback = """
                MATCH (n:Thought)
                WHERE n.session_id = $sid
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
                       n.navigator_insight as navigator_insight,
                       n.semantic_gist as semantic_gist
                ORDER BY n.created_at DESC
                LIMIT 50
                """
                results = self.db.query(q_fallback, {"sid": session_id})
                # Re-sort for formatting
                results = sorted(results, key=lambda x: x.get("created_at", 0))

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
                if len(row) > 19:
                    base_data["semantic_gist"] = row[19]

                processed_data.append(base_data)

        if not processed_data:
            return "No progress rows."

        # --- RTM HIERARCHICAL COMPRESSION ---
        # Split into gist groups (old, compressed) + leaf rows (recent, detailed)
        gist_chunks, leaf_rows = self._build_rtm_tree(processed_data)

        # Merged summaries (pure structural)
        windowed_data = []
        summaries = []

        for chunk in gist_chunks:
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
            summaries.append(self._generate_gist_structural_summary(chunk))

        # Append leaf rows as-is
        for row in leaf_rows:
            windowed_data.append(row)
            summaries.append(self._get_structural_step_summary(row))

        # Build Table with Row Collapsing (Deduplication)
        lines = []
        lines.append(
            "| Time | REPL | T.S | St | Hash | Gestalt (Ψ,📐,Ω,🧭) | Summary (Action & Outcome) | Recall ID |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")

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
            gestalt = f"Ψ(S:{shaky_str} C:{confluence_str} E:{evasion_str} F:{freedom_str}) ‖ 📐(S:{sheaf_str} H0:{h0_rank_str}) ‖ Ω:{omcd_str}"

            nav_insight = row.get("navigator_insight")
            if nav_insight:
                gestalt += f" ‖ 🧭 {nav_insight}"

            # --- ALERT DECORATION ---
            alerts = []
            if isinstance(sheaf, (int, float)) and sheaf > 0.7:
                alerts.append("!! LOOP !!")
            if isinstance(evasion, (int, float)) and evasion < -0.15:
                alerts.append("!! EVASION !!")
            if isinstance(shaky, (int, float)) and shaky < -0.15:
                alerts.append("!! UNCERTAIN !!")
            if isinstance(h0_rank, (int, float)) and h0_rank > 1:
                alerts.append("!! FRAGMENTED !!")

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

    def _generate_gist_structural_summary(self, chunk: List[Dict]) -> str:
        """Structural Gist: Summarizes multiple units using mathematical metadata.

        Preserves semantic bookends (first and last node summaries) so the LLM
        retains conceptual context after compression instead of only seeing counts.
        """
        count = len(chunk)
        total_mdl = sum(float(r.get("compression_gain", 0.0) or 0.0) for r in chunk)
        ops = [r.get("thimac_op", "?") for r in chunk if r.get("thimac_op")]
        op_counts = {op: ops.count(op) for op in set(ops)}
        op_str = ", ".join(
            f"{v}x{k}" for k, v in sorted(op_counts.items(), reverse=True)
        )

        first_concept = self._get_structural_step_summary(chunk[0])[:60] if chunk else ""
        last_concept = (
            self._get_structural_step_summary(chunk[-1])[:60] if len(chunk) > 1 else ""
        )
        if last_concept and last_concept != first_concept:
            concept_trail = f" | Concepts: {first_concept}…→…{last_concept}"
        elif first_concept:
            concept_trail = f" | Concept: {first_concept}"
        else:
            concept_trail = ""

        return f"[Gist: {count} units] ΣMDL: {total_mdl:+.3f} | Ops: {op_str}{concept_trail}"
        # Include most recent semantic gist for higher fidelity
        sample_gist = ""
        for r in reversed(chunk):
            gist = r.get("semantic_gist")
            if gist:
                sample_gist = f" | Gist: {gist[:50]}..."
                break

        return (
            f"[Gist: {count} units] ΣMDL: {total_mdl:+.3f}{sample_gist} | Ops: {op_str}"
        )

    def _get_structural_step_summary(self, row: Dict) -> str:
        """Retrieves the dense semantic gist or pre-computed MDL footprint."""
        gist = row.get("semantic_gist")
        if gist:
            return str(gist)

        summary = row.get("step_summary")
        if summary:
            return str(summary)

        # Fallback if both are missing (e.g. legacy nodes)
        op = row.get("thimac_op") or "PROCESS"
        lvl = row.get("thimac_level") or "SUBSISTENCE"
        gain = float(row.get("compression_gain") or 0.0)
        return f"{op} [{lvl}] (MDL: {gain:+.3f})"

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

            # [UI POLISH] Limit Subject/Relation/Object length to 30 chars
            subject = str(subject)[:30]
            relation = str(relation)[:30]
            target = str(target)[:30]

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
