"""
Based on Al-Fedaghi's "Thinging Machines" (TM) framework.
Replaces the NCA-based MorphologicalMemory with a semantically meaningful
session state tracker using the two-level ontology:

    EXISTENCE  — concrete, materialized results (sensed events)
    SUBSISTENCE — potential reality, knowledge, plans (footprints of events)

Five Thimac operations (Stages) classify every agent action:
    ARRIVE   — input signal, data ingestion, turn initialization
    ACCEPT   — validation success, grounding confirmation, state locking
    PROCESS  — transformation, computation, reasoning, analysis
    RELEASE  — output generation, file writing, final response
    TRANSFER — movement to other machines, MCP calls, sub-REPL delegation
"""

import logging
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("graph_rlm.thimac")


class ThimacOperation(str, Enum):
    """Five official Thinging Machine stages."""

    ARRIVE = "ARRIVE"
    ACCEPT = "ACCEPT"
    PROCESS = "PROCESS"
    RELEASE = "RELEASE"
    TRANSFER = "TRANSFER"


class ThimacLevel(str, Enum):
    """Two-level ontological reality."""

    EXISTENCE = "EXISTENCE"  # Materialized, verified results
    SUBSISTENCE = "SUBSISTENCE"  # Potential, planned, or knowledge state


@dataclass
class ThimacEvent:
    """A single event classified by the Thimac ontology."""

    thought_id: str
    operation: ThimacOperation
    level: ThimacLevel
    status: str
    timestamp: Optional[int] = None  # epoch ms
    summary: str = ""
    turn_id: Optional[int] = None
    step_id: Optional[int] = None
    repl_id: Optional[str] = None
    logical_id: Optional[str] = None
    tool_calls: Optional[List[str]] = None
    compression_gain: float = 0.0


class ThimacMemory:
    """
    Session state tracker using existence/subsistence ontology.

    Dual nature (Thimac): each thought is a MACHINE when it performs
    an operation, and a THING when it is the object of another operation.

    This replaces the NCA MorphologicalMemory grid with a semantically
    meaningful state model that the scratchpad can render as a
    human-readable session overview.
    """

    def __init__(self) -> None:
        self.existence: List[ThimacEvent] = []
        self.subsistence: List[ThimacEvent] = []
        self._all_events: List[ThimacEvent] = []

        # --- STATE TRACKING ---
        self.known_skills: List[str] = []
        self.known_files: List[str] = []
        self.knowledge_horizon: List[str] = []  # Results of last 3 ARRIVE/Accept ops
        self._repo_root: Optional[Path] = None

    def _get_repo_root(self) -> Path:
        """Dynamic resolution of repo root to avoid hardcoding."""
        if self._repo_root:
            return self._repo_root

        file_path = Path(__file__).absolute()
        if "graph_rlm" in str(file_path):
            # .../graph_rlm/backend/src/core/thimac_memory.py
            self._repo_root = file_path.parent.parent.parent.parent.parent
        else:
            self._repo_root = Path.cwd()
        return self._repo_root

    # ------------------------------------------------------------------
    # Public API (matches MorphologicalMemory interface for drop-in use)
    # ------------------------------------------------------------------

    def ingest_thought(
        self, thought: Dict, tool_calls: Optional[List[str]] = None
    ) -> ThimacEvent:
        """
        Classify and store a thought node by its Thimac operation and level.
        Updates state tracking for Skills, Files, and Knowledge Horizon.

        Args:
            thought: Dict with keys from the Thought node
            tool_calls: Optional list of tools actually executed (empirical trace)

        Returns:
            The created ThimacEvent.
        """
        operation = self._classify_operation(thought, tool_calls)
        level = self._classify_level(thought)
        summary = self._extract_summary(thought, tool_calls)

        event = ThimacEvent(
            thought_id=thought.get("id", ""),
            operation=operation,
            level=level,
            status=thought.get("status", "unknown"),
            timestamp=thought.get("created_at"),
            summary=summary,
            turn_id=thought.get("turn_id"),
            step_id=thought.get("step_id"),
            repl_id=thought.get("repl_id"),
            logical_id=thought.get("logical_id"),
            tool_calls=tool_calls,
            compression_gain=thought.get("compression_gain", 0.0),
        )

        self._all_events.append(event)

        if level == ThimacLevel.EXISTENCE:
            self.existence.append(event)
            # Update Knowledge/Material State
            self._update_state_tracking(event, thought)
        else:
            self.subsistence.append(event)

        logger.debug(
            "Thimac: %s %s [%s] — %s",
            event.operation.value,
            event.level.value,
            event.status,
            event.summary[:600],
        )
        return event

    def _update_state_tracking(self, event: ThimacEvent, thought: Dict) -> None:
        """Analyze successful results to update the Known State."""
        result = thought.get("result") or ""
        prompt = thought.get("prompt") or ""
        prompt_lower = prompt.lower()
        repo_root = str(self._get_repo_root())

        # 1. Update Known Skills
        if any(
            k in prompt_lower
            for k in ["agentskills", "list_skills", "save_skill", "rlm.save_skill"]
        ):
            # Try to grab names from agentskills list or save_skill definition
            # e.g., rlm.save_skill('my_skill', code) -> grabs 'my_skill'
            skills = re.findall(r"['\"]([a-zA-Z0-9_]+)['\"]", prompt + " " + result)
            for skill in skills:
                if (
                    skill not in self.known_skills
                    and len(skill) > 3
                    and skill not in ["name", "code", "description"]
                ):
                    self.known_skills.append(skill)

        # 2. Update Known Files
        if any(k in prompt_lower for k in ["create_file", "write", "save", "open("]):
            # Use dynamic repo root for grounding instead of hardcoded home
            # Extract paths from original casing for Linux compatibility
            paths = re.findall(r"(/[a-zA-Z0-9._/-]+)", prompt + " " + result)
            # Use dynamic repo root for grounding and temp dir for safety
            system_tmp = tempfile.gettempdir()
            for path in paths:
                if (
                    repo_root in path or system_tmp in path
                ) and path not in self.known_files:
                    self.known_files.append(path)

        # 3. Update Knowledge Horizon (Last 3 Arrival/Acceptance)
        if event.operation in [ThimacOperation.ARRIVE, ThimacOperation.ACCEPT]:
            self.knowledge_horizon.append(f"{event.operation.value}: {event.summary}")
            if len(self.knowledge_horizon) > 3:
                self.knowledge_horizon.pop(0)

    def get_gestalt_string(self) -> str:
        """
        Human-readable session overview using the two-level ontology.
        Enhanced with Stability Anchor and Lupascian Logic.
        """
        if not self._all_events:
            return "No session events tracked yet."

        lines = []

        # --- Stability Anchor: Bernshteyn-LLL Bound ---
        # p(d+1)^8 <= 2^-15
        # We estimate p (error prob) and d (dependency/complexity) from session stats
        total = len(self._all_events)
        failed = sum(1 for e in self._all_events if e.status != "success")
        p = (failed / total) if total > 0 else 0
        d = min(10, total // 5)  # Heuristic complexity
        bound = p * ((d + 1) ** 8)
        threshold = 2**-15
        stability = "STABLE" if bound <= threshold else "CHAOTIC"

        lines.append(f"### 🌐 Cog-State: {stability}")
        lines.append(f"**Math Anchor**: $p(d+1)^8 = {bound:.6f} \\le 2^{{-15}}$")
        lines.append("")

        # --- STATE TRACKING: Known Environment ---
        lines.append("### 🧠 Current Known State")
        skills_str = (
            ", ".join(self.known_skills[-10:]) if self.known_skills else "None yet"
        )
        lines.append(f"- **Known Skills**: [{skills_str}]")

        last_file = self.known_files[-1] if self.known_files else "None"
        lines.append(f"- **Last File Action**: `{last_file}`")

        horizon = (
            " | ".join(self.knowledge_horizon) if self.knowledge_horizon else "Clear"
        )
        lines.append(f"- **Arrival Horizon**: {horizon}")

        # Lupascian Logic: Negative Events (Formal absence as state)
        if self.subsistence:
            neg_events = [
                s.summary
                for s in self.subsistence
                if s.status in ["failed", "rejected"]
            ][-3:]
            if neg_events:
                lines.append(
                    f"- **Negative States (Lupascian)**: {', '.join(neg_events)}"
                )

        lines.append("")

        # --- EXISTENCE: What has actually materialized ---
        if self.existence:
            lines.append("**Existence** (Materialized Results):")
            for e in self.existence[-3:]:
                ts_str = self._format_ts(e.timestamp)
                lines.append(f"  {e.operation.value}: {e.summary} [{ts_str}]")
        else:
            lines.append("**Existence**: No materialized results yet.")

        # --- SUBSISTENCE: Knowledge / potential ---
        if self.subsistence:
            lines.append("**Subsistence** (Potential/Footprints):")
            for s in self.subsistence[-3:]:
                ts_str = self._format_ts(s.timestamp)
                lines.append(f"  {s.operation.value}: {s.summary} [{ts_str}]")
        else:
            lines.append("**Subsistence**: No knowledge state captured yet.")

        # --- Counts ---
        exist_pct = (len(self.existence) / total * 100) if total else 0
        lines.append(
            f"\n*{total} total events | "
            f"{exist_pct:.0f}% existence | "
            f"{100 - exist_pct:.0f}% subsistence*"
        )

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all tracked events (new session)."""
        self.existence.clear()
        self.subsistence.clear()
        self._all_events.clear()

    def adapt_to_stress(self, topological_stress: float) -> None:
        """
        Dynamically compresses the session memory if topological stress is high.
        Aggressively drops 'PROCESS' nodes and trims the history buffers to prevent
        the context window from filling with noisy dead-ends.
        """
        if topological_stress < 0.15:
            # Baseline maintenance: keep arrays from ballooning indefinitely
            if len(self.existence) > 40:
                self.existence = self.existence[-40:]
            if len(self.subsistence) > 40:
                self.subsistence = self.subsistence[-40:]
            if len(self._all_events) > 100:
                self._all_events = self._all_events[-100:]
            return

        logger.info(
            "Thimac: High stress detected (%.2f). Compressing memory...",
            topological_stress,
        )

        # High Stress Compression:
        # 1. Keep only the last 10 'EXISTENCE' (hard facts)
        self.existence = self.existence[-10:]

        # 2. Aggressively filter 'SUBSISTENCE' (thoughts/plans)
        # Keep only ACCEPT/ARRIVE operations, drop speculative PROCESS loops.
        filtered_sub = [
            s
            for s in self.subsistence
            if s.operation in [ThimacOperation.ACCEPT, ThimacOperation.ARRIVE]
        ]
        # Always keep the very last 2 nodes regardless of type so we don't lose immediate context
        last_two = (
            self.subsistence[-2:] if len(self.subsistence) >= 2 else self.subsistence
        )

        # Merge and deduplicate (preserving order)
        new_sub = []
        seen_ids = set()

        # Phase 4 Gestalt: Persistent Homology
        # Nodes that mathematically compress the state space (MDL gain > 0)
        # survive the pruning wave natively, becoming formalized persistent structures.
        persistent_homology_nodes = [
            s
            for s in self.subsistence
            if s.compression_gain > 0.1 and s not in filtered_sub
        ]

        for s in filtered_sub + persistent_homology_nodes + last_two:
            if s.thought_id not in seen_ids:
                new_sub.append(s)
                seen_ids.add(s.thought_id)

        self.subsistence = new_sub[-15:]

        # Sync master list
        all_kept_ids = {e.thought_id for e in self.existence + self.subsistence}
        self._all_events = [e for e in self._all_events if e.thought_id in all_kept_ids]

    # ------------------------------------------------------------------
    # Classification Logic
    # ------------------------------------------------------------------

    def _classify_operation(
        self, thought: Dict, tool_calls: Optional[List[str]] = None
    ) -> ThimacOperation:
        """
        Classify the Thimac operation based on empirical tool calls (priority)
        or keyword heuristics (fallback).
        """
        # 0. Empirical Detection (Priority)
        if tool_calls:
            # Map tool names to operations
            for tool in tool_calls:
                t = tool.lower()
                # ARRIVE: Reading/Searching
                if any(
                    k in t
                    for k in [
                        "recall",
                        "search",
                        "history",
                        "read",
                        "list",
                        "view",
                        "grep",
                        "fd",
                        "ls",
                    ]
                ):
                    return ThimacOperation.ARRIVE
                # RELEASE: Externalizing/Saving
                if any(
                    k in t for k in ["save", "done", "write", "create", "notify_user"]
                ):
                    return ThimacOperation.RELEASE
                # TRANSFER: Moving context/IPC
                if any(
                    k in t for k in ["run_skill", "delegate", "ipc", "repl", "call"]
                ):
                    return ThimacOperation.TRANSFER

            # If tools were called but didn't match specific categories, it's a PROCESS
            return ThimacOperation.PROCESS

        # 1. Keyword Heuristics (Fallback)
        prompt = (thought.get("prompt") or "").lower()
        result = (thought.get("result") or "").lower()
        status = (thought.get("status") or "").lower()
        combined = prompt + " " + result

        # 1. TRANSFER: Movement to other machines (Delegation/MCP)
        if any(
            k in combined
            for k in ["sub_repl", "transfer", "delegate", "mcp", "ipc", "repl_id"]
        ):
            return ThimacOperation.TRANSFER

        # 2. RELEASE: Output generation or state externalization
        if any(
            k in combined
            for k in ["final_response", "write(", "save(", "create_file", "notify_user"]
        ):
            return ThimacOperation.RELEASE

        # 3. ACCEPT: Validation success or grounding
        if status == "success" and (
            not result
            or any(k in combined for k in ["verified", "grounded", "confirmed"])
        ):
            return ThimacOperation.ACCEPT

        # 4. ARRIVE: Ingestion (Initial thought or search result)
        if any(
            k in combined
            for k in [
                "read(",
                "fetch",
                "search",
                "grep",
                "cat ",
                "view_",
                "ls ",
                "list_dir",
                "find",
                "fd ",
                "glob",
                "locate",
                "stat",
                "check_path",
            ]
        ):
            return ThimacOperation.ARRIVE

        # 5. PROCESS: Transformation/Reasoning (Default)
        return ThimacOperation.PROCESS

    def _classify_level(self, thought: Dict) -> ThimacLevel:
        """
        Classify whether a thought has materialized (existence) or
        remains in potential/knowledge state (subsistence).

        Existence = verified concrete result.
        Subsistence = plan, knowledge, failed attempt, pending.
        """
        status = (thought.get("status") or "").lower()

        # Concrete, verified results → EXISTENCE
        if status == "success":
            return ThimacLevel.EXISTENCE

        # Everything else → SUBSISTENCE (potential/knowledge)
        # "failed" = learned what doesn't work (footprint of event)
        # "pending" = planned but not yet materialized
        # "rejected" = dreamer invalidated (subsists as knowledge)
        # "wake" = system intervention (subsists as context)
        return ThimacLevel.SUBSISTENCE

    def _extract_summary(
        self, thought: Dict, tool_calls: Optional[List[str]] = None
    ) -> str:
        """
        Extracts a purely topological footprint of the event.
        Replaces LLM text lossy summaries with structural MDL graphs.
        """
        status = (thought.get("status") or "").lower()
        compression = thought.get("compression_gain", 0.0)

        # Base mathematical identifier (UUID prefix)
        footprint = f"[{thought.get('id', 'N/A')[:8]}] "

        # Add Minimum Description Length (MDL) metric
        if abs(compression) > 0.01:
            footprint += f"(MDL: {compression:+.2f}) "

        # Add Execution Edge Types
        if tool_calls:
            main_tool = tool_calls[0]
            if len(tool_calls) > 1:
                footprint += f"Edge: {main_tool} (+{len(tool_calls)-1} more)"
            else:
                footprint += f"Edge: {main_tool}"
        elif status == "success":
            footprint += "Edge: Axiomatic_Transform"
        elif status in ["failed", "error", "rejected"]:
            footprint += "Edge: Broken_Sympathy"
        else:
            footprint += "Edge: Latent_Vector"

        return footprint

    @staticmethod
    def _format_ts(epoch_ms: Optional[int]) -> str:
        """Format epoch milliseconds to HH:MM:SS.mmm."""
        if not epoch_ms:
            return "--:--:--.---"
        try:
            ts = datetime.fromtimestamp(epoch_ms / 1000)
            return ts.strftime("%H:%M:%S.") + f"{ts.microsecond // 1000:03d}"
        except (OSError, ValueError, OverflowError):
            return "--:--:--.---"
