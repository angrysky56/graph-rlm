"""
Thimac Memory System — Session State via Existence/Subsistence Ontology.

Based on Al-Fedaghi's "Thinging Machines" (TM) framework.
Replaces the NCA-based MorphologicalMemory with a semantically meaningful
session state tracker using the two-level ontology:

    EXISTENCE  — concrete, materialized results (sensed events)
    SUBSISTENCE — potential reality, knowledge, plans (footprints of events)

Five Thimac operations classify every agent action:
    CREATE   — file write, code generation, resource allocation
    PROCESS  — code execution, analysis, computation
    RECEIVE  — input ingestion, API response, data retrieval
    TRANSFER — data movement, IPC, MCP calls, sub-REPL delegation
    RELEASE  — output delivery, final response, file save confirmation
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger("graph_rlm.thimac")


class ThimacOperation(str, Enum):
    """Five fundamental Thimac operations."""

    CREATE = "CREATE"
    PROCESS = "PROCESS"
    RECEIVE = "RECEIVE"
    TRANSFER = "TRANSFER"
    RELEASE = "RELEASE"


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

    # ------------------------------------------------------------------
    # Public API (matches MorphologicalMemory interface for drop-in use)
    # ------------------------------------------------------------------

    def ingest_thought(self, thought: Dict) -> ThimacEvent:
        """
        Classify and store a thought node by its Thimac operation and level.

        Args:
            thought: Dict with keys from the Thought node
                     (id, prompt, status, result, created_at, repl_id,
                      turn_id, step_id)

        Returns:
            The created ThimacEvent.
        """
        operation = self._classify_operation(thought)
        level = self._classify_level(thought)
        summary = self._extract_summary(thought)

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
        )

        self._all_events.append(event)

        if level == ThimacLevel.EXISTENCE:
            self.existence.append(event)
        else:
            self.subsistence.append(event)

        logger.debug(
            "Thimac: %s %s [%s] — %s",
            event.operation.value,
            event.level.value,
            event.status,
            event.summary[:60],
        )
        return event

    def get_gestalt_string(self) -> str:
        """
        Human-readable session overview using the two-level ontology.
        Injected into the scratchpad as the Morphological Gestalt section.
        """
        if not self._all_events:
            return "No session events tracked yet."

        lines = []

        # --- EXISTENCE: What has actually materialized ---
        if self.existence:
            lines.append(f"**Existence** ({len(self.existence)} events):")
            for e in self.existence[-5:]:
                ts_str = self._format_ts(e.timestamp)
                lines.append(f"  {e.operation.value}: {e.summary} [{ts_str}]")
        else:
            lines.append("**Existence**: No materialized results yet.")

        # --- SUBSISTENCE: Knowledge / potential ---
        if self.subsistence:
            lines.append(f"**Subsistence** ({len(self.subsistence)} items):")
            for s in self.subsistence[-5:]:
                ts_str = self._format_ts(s.timestamp)
                lines.append(f"  {s.operation.value}: {s.summary} [{ts_str}]")
        else:
            lines.append("**Subsistence**: No knowledge state captured yet.")

        # --- Counts ---
        total = len(self._all_events)
        exist_pct = (len(self.existence) / total * 100) if total else 0
        lines.append(
            f"*{total} total events | "
            f"{exist_pct:.0f}% materialized | "
            f"{100 - exist_pct:.0f}% potential*"
        )

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all tracked events (new session)."""
        self.existence.clear()
        self.subsistence.clear()
        self._all_events.clear()

    # ------------------------------------------------------------------
    # Classification Logic
    # ------------------------------------------------------------------

    def _classify_operation(self, thought: Dict) -> ThimacOperation:
        """
        Classify the thought's primary Thimac operation based on its content.
        """
        prompt = (thought.get("prompt") or "").lower()
        result = (thought.get("result") or "").lower()
        combined = prompt + " " + result

        # RELEASE: Final output delivery
        if any(
            k in combined
            for k in ["rlm_final_output", "final_response", "task completed"]
        ):
            return ThimacOperation.RELEASE

        # CREATE: File/resource creation
        if any(
            k in combined
            for k in [
                "open(",
                "write(",
                "makedirs",
                "mkdir",
                "save(",
                "create_file",
                "with open",
            ]
        ):
            return ThimacOperation.CREATE

        # TRANSFER: Data movement / delegation
        if any(
            k in combined
            for k in [
                "sub_repl",
                "transfer",
                "delegate",
                "mcp",
                "ipc",
                "spawn",
            ]
        ):
            return ThimacOperation.TRANSFER

        # RECEIVE: Input / data retrieval
        if any(
            k in combined
            for k in [
                "read(",
                "input",
                "fetch",
                "api",
                "response",
                "recv",
                "get(",
                "load(",
            ]
        ):
            return ThimacOperation.RECEIVE

        # PROCESS: Default — execution / analysis
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

    def _extract_summary(self, thought: Dict) -> str:
        """
        Extract a brief summary from the thought's content.
        Prefers execution_summary, then first line of result, then prompt.
        """
        # Try execution_summary first (if populated from earlier runs)
        es = thought.get("execution_summary")
        if es and len(str(es).strip()) > 5:
            return str(es).strip()[:80]

        # Try first meaningful line of result
        result = thought.get("result") or ""
        if result:
            first_line = result.strip().split("\n")[0][:80]
            if len(first_line) > 5:
                return first_line

        # Fall back to prompt
        prompt = thought.get("prompt") or ""
        first_line = prompt.strip().split("\n")[0][:80]
        return first_line if first_line else "(empty)"

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
