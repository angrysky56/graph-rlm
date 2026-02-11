"""
Shared execution state and tracing utilities for the Graph-RLM Agent.
"""

import contextvars
import queue
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional

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

            # Route messages based on content
            ui_target = "TERMINAL_RAW"
            if any(
                k in clean_msg
                for k in ["[THINKING]", "Axiomatic", "Reflexion"]
            ):
                ui_target = "CHAT_RESPONSE"

            q.put_nowait(
                {"type": "trace", "content": clean_msg, "ui_target": ui_target}
            )
    except LookupError:
        pass
    except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
        # Fallback log to avoid recursion loop if logging fails
        sys.stderr.write(f"Failed to broadcast trace: {e}\n")
