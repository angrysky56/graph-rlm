import contextvars
from dataclasses import dataclass
from typing import Optional

# Session-specific state isolated by thread/context
agent_state: contextvars.ContextVar[Optional["ExecutionState"]] = contextvars.ContextVar(
    "agent_state", default=None
)

@dataclass
class ExecutionState:
    final_result: Optional[str] = None
    stop_requested: bool = False
    synthesis_triggered: bool = False
    current_thought_id: Optional[str] = None
    depth: int = 0
