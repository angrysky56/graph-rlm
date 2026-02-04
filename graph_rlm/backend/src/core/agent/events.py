import contextvars
import queue
import re
import sys
import threading
from typing import Optional, Any
from ..logger import get_logger

logger = get_logger("graph_rlm.agent.events")

# Context Variable to hold the event queue for the current execution thread/chain
execution_events: contextvars.ContextVar[Optional[queue.Queue]] = (
    contextvars.ContextVar("execution_events", default=None)
)

def broadcast_trace(msg: str):
    """Monitor callback to push trace logs to the active event loop."""
    try:
        q = execution_events.get()
        if q:
            # Clean ANSI codes for UI (optional, but UI handles raw text better usually)
            clean_msg = re.sub(r"\x1b\[[0-9;]*m", "", msg)
            q.put_nowait({"type": "trace", "content": clean_msg})
    except LookupError:
        pass
    except Exception as e:
        # Fallback log to avoid recursion loop if logging fails
        sys.stderr.write(f"Failed to broadcast trace: {e}\n")

class EventEmitter:
    def __init__(self):
        pass

    def emit(
        self,
        event_type: str,
        data: Any = None,
        content: Optional[str] = None,
        code: Optional[str] = None,
        is_sub_event: bool = False,
        tag: Optional[str] = None,
    ):
        """
        Helper to emit events to the current context's queue if it exists.
        Also mirrors key events to the server logs (terminal) for visibility.
        """
        prefix = "↳ " if is_sub_event else ""

        # Mirror to Terminal/Logs
        if event_type == "thinking" and content:
            # Use tag if available for better log mirroring
            log_prefix = f"[THINKING] [{tag}]" if tag else "[THINKING]"
            logger.info(f"{prefix}{log_prefix} {content.strip()}")
        elif event_type == "code_output" and content:
            logger.info(f"{prefix}[REPL OUTPUT] >>\n{content}")
        elif event_type == "error" and content:
            logger.error(f"{prefix}[AGENT ERROR] {content}")

        q = execution_events.get()
        if q:
            payload = {"type": event_type, "is_sub_event": is_sub_event}
            if data:
                payload["data"] = data
            if content:
                # Automate REPL ID prefixing for code output if not present
                payload["content"] = f"{prefix}{content}"
            if code:
                payload["code"] = code
            if tag:
                payload["tag"] = tag
            q.put(payload)
