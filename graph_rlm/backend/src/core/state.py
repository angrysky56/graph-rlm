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
        if q:
            # Simple approach: Strip ANSI for cleaner UI text
            clean_msg = re.sub(r"\x1b\[[0-9;]*m", "", msg)

            # --- Signal vs. Noise Filter ---
            # 1. Block List (Internal Implementation Details)
            block_list = ["[LLM]", "[DB]", "[REPL]", "[SHEAF]", "[TRACE]"]
            if any(tag in clean_msg for tag in block_list):
                q.put_nowait(
                    {"type": "trace", "content": clean_msg, "ui_target": "TERMINAL_RAW"}
                )
                return

            # 2. Structured Logic Components
            # These map to specific UI "boxes" for better UX
            ui_target = "TERMINAL_RAW"
            ui_component = "text"

            if "[META]" in clean_msg:
                ui_target = "CHAT_RESPONSE"
                ui_component = "meta_box"
            elif "[REFLEXION]" in clean_msg:
                ui_target = "CHAT_RESPONSE"
                ui_component = "reflexion_box"
            elif "[SKILL]" in clean_msg:
                ui_target = "CHAT_RESPONSE"
                ui_component = "skill_box"
            elif "[DREAMER]" in clean_msg:
                ui_target = "CHAT_RESPONSE"
                ui_component = "dreamer_box"
            elif "[AGENT]" in clean_msg:
                # Agent messages must be high-signal to be shown
                if any(
                    x in clean_msg
                    for x in ["Plan:", "Action:", "Decision:", "Final Answer"]
                ):
                    ui_target = "CHAT_RESPONSE"
                    ui_component = "text"
            elif "RLM_FINAL_OUTPUT" in clean_msg:
                ui_target = "CHAT_RESPONSE"
                ui_component = "final_result"

            q.put_nowait(
                {
                    "type": "trace",
                    "content": clean_msg,
                    "ui_target": ui_target,
                    "ui_component": ui_component,
                }
            )
    except LookupError:
        pass
    except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
        # Fallback log to avoid recursion loop if logging fails
        sys.stderr.write(f"Failed to broadcast trace: {e}\n")
