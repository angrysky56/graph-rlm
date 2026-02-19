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
    """Thread-local state for the agent's execution loop."""

    final_result: Optional[str] = None
    stop_requested: bool = False
    synthesis_triggered: bool = False
    current_thought_id: Optional[str] = None
    depth: int = 0
    turn_id: int = 1
    recursion_stack: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


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
