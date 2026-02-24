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
from typing import Dict, List, Optional, Tuple

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


class ThimacIntention(str, Enum):
    """LIDA-inspired hierarchy of intentions."""

    DISTAL = "DISTAL"  # Long-term goal, agential agenda
    PROXIMAL = "PROXIMAL"  # Readiness for action, current step plan
    MOTOR = "MOTOR"  # Executory action, REPL command


@dataclass
class ThimacEvent:
    """A single event classified by the Thimac ontology."""

    thought_id: str
    operation: ThimacOperation
    level: ThimacLevel
    status: str
    operation_reason: str = ""
    level_reason: str = ""
    full_data: str = ""  # Untruncated prompt + result
    timestamp: Optional[int] = None  # epoch ms
    summary: str = ""
    semantic_gist: str = ""
    turn_id: Optional[int] = None
    step_id: Optional[int] = None
    repl_id: Optional[str] = None
    logical_id: Optional[str] = None
    tool_calls: Optional[List[str]] = None
    compression_gain: float = 0.0
    is_branching: bool = False
    intent_type: ThimacIntention = ThimacIntention.MOTOR


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
        self,
        thought: Dict,
        tool_calls: Optional[List[str]] = None,
        is_branching: bool = False,
        semantic_gist: str = "",
        intent_type: Optional[ThimacIntention] = None,
    ) -> ThimacEvent:
        """Classifies a thought and updates session state."""
        op, op_reason = self._classify_operation(thought, tool_calls)
        lvl, lvl_reason = self._classify_level(thought)
        summary = self._extract_summary(thought, tool_calls)
        full_data = f"{thought.get('prompt', '')}\n{thought.get('result', '')}"

        event = ThimacEvent(
            thought_id=thought.get("id", ""),
            operation=op,
            level=lvl,
            status=thought.get("status", "unknown"),
            operation_reason=op_reason,
            level_reason=lvl_reason,
            full_data=full_data,
            timestamp=thought.get("created_at"),
            summary=summary,
            semantic_gist=semantic_gist or summary,
            turn_id=thought.get("turn_id"),
            step_id=thought.get("step_id"),
            repl_id=thought.get("repl_id"),
            logical_id=thought.get("logical_id"),
            tool_calls=tool_calls,
            compression_gain=thought.get("compression_gain", 0.0),
            is_branching=is_branching,
            intent_type=intent_type or self._align_intent(op, lvl, tool_calls),
        )

        self._all_events.append(event)

        if lvl == ThimacLevel.EXISTENCE:
            self.existence.append(event)
            # Update Knowledge/Material State
            self._update_state_tracking(event, thought)
        else:
            self.subsistence.append(event)

        logger.debug(
            "Thimac: %s (%s) %s (%s) [%s] — %s",
            event.operation.value,
            event.operation_reason,
            event.level.value,
            event.level_reason,
            event.status,
            event.summary[:600],
        )
        return event

    def _align_intent(
        self, op: ThimacOperation, lvl: ThimacLevel, tool_calls: Optional[List[str]]
    ) -> ThimacIntention:
        """Aligns Thimac Operation with LIDA Intention hierarchy."""
        if op == ThimacOperation.ACCEPT:
            # Acceptance of a goal or validation is a Distal alignment
            return ThimacIntention.DISTAL
        if tool_calls or op == ThimacOperation.TRANSFER:
            # Direct action in the world
            return ThimacIntention.MOTOR
        if op == ThimacOperation.RELEASE and lvl == ThimacLevel.EXISTENCE:
            # Final delivery
            return ThimacIntention.DISTAL
        if op == ThimacOperation.PROCESS or op == ThimacOperation.ARRIVE:
            # Reasoning or ingestion for planning
            return ThimacIntention.PROXIMAL
        return ThimacIntention.PROXIMAL

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
        Human-readable session overview using pure Structural Information Theory.
        Replaces LLM-based text summaries with MDL-grounded metrics.
        """
        if not self._all_events:
            return "No session events tracked yet."

        lines = []

        # --- Stability Anchor: Bernshteyn-LLL Bound ---
        total = len(self._all_events)
        failed = sum(1 for e in self._all_events if e.status != "success")
        p = (failed / total) if total > 0 else 0
        d = min(10, total // 5)  # Heuristic complexity
        bound = p * ((d + 1) ** 8)
        threshold = 2**-15
        stability = "STABLE" if bound <= threshold else "CHAOTIC"

        # MDL Entropy (Structural Information)
        avg_gain = (
            sum(e.compression_gain for e in self._all_events) / total
            if total > 0
            else 0.0
        )

        lines.append(f"### 🌐 Cog-State: {stability} (MDL Gain: {avg_gain:+.3f})")

        # Branching awareness
        recent_branching = any(e.is_branching for e in self._all_events[-5:])
        if recent_branching:
            lines.append("> [!WARNING]")
            lines.append(
                "> **TOPOLOGICAL BRANCHING DETECTED**: The state space is currently destabilized (sensitive). Strategy shift likely."
            )

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

        # --- EXISTENCE & SUBSISTENCE GESTALT ---
        # Instead of listing summaries (which are now footprints), we show the structural distribution
        e_ops = [e.operation.value for e in self.existence[-10:]]
        s_ops = [s.operation.value for s in self.subsistence[-10:]]

        lines.append(
            f"- **Existence Ops (Recency)**: {', '.join(e_ops) if e_ops else 'None'}"
        )
        lines.append(
            f"- **Subsistence Ops (Recency)**: {', '.join(s_ops) if s_ops else 'None'}"
        )

        # Persistent Homology: Nodes survived pruning cycles
        survivors = [
            f"{s.thought_id[:8]} ({s.intent_type.value})"
            for s in self.subsistence
            if s.compression_gain > 0.15
        ]
        if survivors:
            lines.append(
                f"- **Persistent Homology (Stable Clusters)**: {', '.join(survivors)}"
            )

        # --- SEMANTIC GROUNDING: Directive Gists ---
        # Show recent semantic gists to provide "Directive" context for the agent
        recent_events = self._all_events[-3:]
        if recent_events:
            lines.append("\n### 📜 Recent Directive Gists")
            for e in recent_events:
                # Reconcile math (operation) with semantics (gist)
                intent_info = f" | Intent: {e.intent_type.value}"
                reason_info = f" ({e.operation_reason})"
                lines.append(
                    f"- **{e.operation.value}**{reason_info}: {e.semantic_gist}{intent_info}"
                )

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all tracked events (new session)."""
        self.existence.clear()
        self.subsistence.clear()
        self._all_events.clear()

    def adapt_to_stress(self, topological_stress: float) -> None:
        """
        Dynamically compresses the session memory using Persistent Homology.
        Nodes that mathematically compress the state space survive pruning rounds.
        """
        if topological_stress < 0.15:
            # Baseline maintenance
            if len(self.existence) > 30:
                self.existence = self.existence[-30:]
            if len(self.subsistence) > 30:
                self.subsistence = self.subsistence[-30:]
            return

        logger.info(
            "Thimac: High stress (%.2f). Pruning non-persistent structures...",
            topological_stress,
        )

        # 1. EXISTENCE: Keep last 10 hard facts
        self.existence = self.existence[-10:]

        # 2. SUBSISTENCE (Persistent Homology):
        # We define survival based on MDL gain. Nodes with high gain represent
        # structural 'Closure' and are preserved as persistent homology.
        persistent = [s for s in self.subsistence if s.compression_gain > 0.1]

        # We also keep recent ARRIVE/ACCEPT nodes for grounding
        grounding = [
            s
            for s in self.subsistence
            if s.operation in [ThimacOperation.ACCEPT, ThimacOperation.ARRIVE]
        ]

        # Always keep the absolute last 2 steps
        latest = (
            self.subsistence[-2:] if len(self.subsistence) >= 2 else self.subsistence
        )

        # Merge, deduplicate, and sort by original timestamp (implied by insertion order)
        seen_ids = set()
        new_sub = []
        for s in persistent + grounding + latest:
            if s.thought_id not in seen_ids:
                new_sub.append(s)
                seen_ids.add(s.thought_id)

        self.subsistence = new_sub[-15:]

        # Sync all_events
        kept_ids = {e.thought_id for e in self.existence + self.subsistence}
        self._all_events = [e for e in self._all_events if e.thought_id in kept_ids]

    # ------------------------------------------------------------------
    # Classification Logic
    # ------------------------------------------------------------------

    def _classify_operation(
        self, thought: Dict, tool_calls: Optional[List[str]] = None
    ) -> Tuple[ThimacOperation, str]:
        """
        Classify the Thimac operation and provide an objective reason.
        """
        # 0. Empirical Detection (Priority)
        if tool_calls:
            for tool in tool_calls:
                t = tool.lower()
                if any(
                    k in t for k in ["recall", "search", "read", "view", "ls", "grep"]
                ):
                    return (
                        ThimacOperation.ARRIVE,
                        f"Triggered by ingestion tool: {tool}",
                    )
                if any(k in t for k in ["save", "done", "write", "notify"]):
                    return (
                        ThimacOperation.RELEASE,
                        f"Triggered by externalization tool: {tool}",
                    )
                if any(k in t for k in ["run_skill", "delegate", "ipc", "repl"]):
                    return (
                        ThimacOperation.TRANSFER,
                        f"Triggered by delegation tool: {tool}",
                    )

            return ThimacOperation.PROCESS, "Tools called for transformation/analysis"

        # 1. Keyword Heuristics (Fallback)
        prompt = (thought.get("prompt") or "").lower()
        result = (thought.get("result") or "").lower()
        status = (thought.get("status") or "").lower()
        combined = prompt + " " + result

        if any(
            k in combined
            for k in ["sub_repl", "transfer", "delegate", "mcp", "ipc", "repl_id"]
        ):
            return ThimacOperation.TRANSFER, "Heuristic: Delegation keywords detected"

        # 2. RELEASE: Output generation or state externalization
        if any(
            k in combined
            for k in ["final_response", "write(", "save(", "create_file", "notify_user"]
        ):
            return (
                ThimacOperation.RELEASE,
                "Heuristic: Externalization keywords detected",
            )

        # 3. ACCEPT: Validation success or grounding
        if status == "success" and (
            not result
            or any(k in combined for k in ["verified", "grounded", "confirmed"])
        ):
            return ThimacOperation.ACCEPT, "Heuristic: Grounding verification detected"

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
            return ThimacOperation.ARRIVE, "Heuristic: Ingestion keywords detected"

        # 5. PROCESS: Transformation/Reasoning (Default)
        return ThimacOperation.PROCESS, "Default: Inner reasoning/transformation"

    def _classify_level(self, thought: Dict) -> Tuple[ThimacLevel, str]:
        """
        Classify whether a thought has materialized (existence) or
        remains in potential/knowledge state (subsistence).
        """
        status = (thought.get("status") or "").lower()
        result = thought.get("result")

        # Concrete, verified results → EXISTENCE
        if status == "success" and result:
            return ThimacLevel.EXISTENCE, "Result materialized in execution environment"

        if status == "success":
            return ThimacLevel.EXISTENCE, "Action confirmed as completed (Axiomatic)"

        # Everything else → SUBSISTENCE
        return ThimacLevel.SUBSISTENCE, "Thought remains in latent/potential state"

    def _extract_summary(
        self, thought: Dict, tool_calls: Optional[List[str]] = None
    ) -> str:
        """
        Extracts a purely topological footprint of the event.
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
