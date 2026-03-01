"""
Shared execution state and tracing utilities for the Graph-RLM Agent.
"""

import contextvars
import queue
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Context variable for the event queue
execution_events: contextvars.ContextVar[Optional[queue.Queue]] = (
    contextvars.ContextVar("execution_events", default=None)
)


@dataclass
class ExecutionState:
    """Thread-local state for the agent's execution loop.

    Extended with phase/momentum tracking (Phase C1) so the stateless agent
    can know where it has been and where it needs to go step by step.
    """

    # --- Core State ---
    final_result: Optional[str] = None
    stop_requested: bool = False
    synthesis_triggered: bool = False
    current_thought_id: Optional[str] = None
    round_id: Optional[str] = None
    depth: int = 0
    turn_id: int = 1
    recursion_stack: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --- Phase Awareness (C1) ---
    phase: str = "EXPLORING"  # EXPLORING | EXECUTING | VALIDATING | SYNTHESIZING
    branching_state: str = "STABLE"  # STABLE | BRANCHING
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    intervention_count: int = 0
    consecutive_interventions: int = 0
    tools_used_this_turn: List[str] = field(default_factory=list)
    last_dreamer_critique: Optional[str] = None
    pending_side_effects: List[str] = field(
        default_factory=list
    )  # Trails of unverified file writes

    # --- Momentum Tracking (C1) ---
    # Ring buffer of last N step outcomes for quick trajectory assessment
    step_outcomes: List[str] = field(default_factory=list)

    # --- Monitor Snapshots (C1) ---
    last_sheaf_energy: float = 0.0
    last_sheaf_rationale: Optional[str] = None
    last_omcd_qstop: float = 0.0
    last_omcd_rationale: Optional[str] = None
    last_h0_rank: int = 1

    # --- Cerebellum: Error Pattern Tracking ---
    # Accumulates error types across steps for recurring pattern detection
    error_counts: Dict[str, int] = field(default_factory=dict)

    # --- Thermodynamic State (Phase 8 TLTG) ---
    inference_pressure: float = 0.2
    relational_gravity: float = 0.8
    epistemic_eros: float = 0.5
    free_energy: float = 0.4
    metabolic_state: str = "THETA"


# Session-specific state isolated by thread/context
agent_state: contextvars.ContextVar[Optional[ExecutionState]] = contextvars.ContextVar(
    "agent_state", default=None
)


def broadcast_trace(msg: str):
    """Monitor callback to push trace logs to the active event loop."""
    try:
        q = execution_events.get()
        if not q:
            return

        # @BoundaryStress: Guard against null or non-string messages
        if msg is None:
            return
        if not isinstance(msg, str):
            msg = str(msg)

        # Strip ANSI for cleaner UI text
        clean_msg = re.sub(r"\x1b\[[0-9;]*m", "", msg)

        # --- Signal Routing ---
        ui_target = "TERMINAL_RAW"
        ui_component = "text"

        # Declarative routing map for predictable tag resolution
        routing_map = {
            "[META]": "meta_box",
            "[REFLEXION]": "reflexion_box",
            "[SHEAF]": "sheaf_box",
            "[NAVIGATOR]": "navigator_box",
            "[SKILL]": "skill_box",
            "[DREAMER]": "dreamer_box",
            "RLM_FINAL_OUTPUT": "final_result",
        }

        # Apply routing map
        for tag, comp in routing_map.items():
            if tag in clean_msg:
                ui_target = "CHAT_RESPONSE"
                ui_component = comp
                break

        # Contextual logic for AGENT output
        if ui_target == "TERMINAL_RAW" and "[AGENT]" in clean_msg:
            # Agent messages must be high-signal to be shown
            if any(
                x in clean_msg
                for x in ["Plan:", "Action:", "Decision:", "Final Answer"]
            ):
                ui_target = "CHAT_RESPONSE"
                ui_component = "text"

        payload = {"type": "trace", "content": clean_msg, "ui_target": ui_target}

        # Only inject the exact UI Component if required by the client mapping
        if ui_target == "CHAT_RESPONSE":
            payload["ui_component"] = ui_component
        elif ui_target == "TERMINAL_RAW":
            internal_tags = ["[LLM]", "[DB]", "[REPL]", "[TRACE]"]
            if not any(tag in clean_msg for tag in internal_tags):
                payload["ui_component"] = ui_component

        import asyncio

        # @BoundaryStress: Fast queue puts can hit limits (QueueFull) - swallow drop rather than crash agent
        try:
            q.put_nowait(payload)
        except (queue.Full, asyncio.QueueFull):
            pass

    except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
        sys.stderr.write(f"Failed to broadcast trace: {e}\n")
