"""
Observability and Tracing layer for Graph-RLM.
Provides high-fidelity logging of agent actions and system state transitions.
"""

import logging
import sys
from typing import Any, Optional

# ═══════════════════════════════════════════════════════════════
# RLM Pipeline Triggers (Canonical Constants)
# ═══════════════════════════════════════════════════════════════
# Agent emits at start after profiling task complexity/persona.
RLM_AGENT_TASK_PLAN = "RLM_AGENT_TASK_PLAN"
# Agent proposes a candidate answer (NOT a stop signal).
RLM_INITIAL_RESPONSE = "RLM_INITIAL_RESPONSE"
# Dreamer found issues during validation — agent must fix them.
RLM_DREAMER_ISSUES = "RLM_DREAMER_ISSUES"
# Dreamer validated the candidate — agent may write final report.
RLM_DREAMER_VALIDATED = "RLM_DREAMER_VALIDATED"
# Agent writes final report/artifact — this is the ONLY agent-side stop signal.
RLM_FINAL_OUTPUT = "RLM_FINAL_OUTPUT"

# ANSI Colors
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

logger = logging.getLogger("graph_rlm.trace")


class _TraceState:
    """Internal container for tracing state to avoid global statements."""

    def __init__(self):
        self.monitor_callback = None


_state = _TraceState()


def register_monitor(callback):
    """Registers a callback(msg: str) to receive trace logs in real-time."""
    _state.monitor_callback = callback


def trace_action(
    context: str,
    action: str,
    result: Optional[Any] = None,
    level: str = "info",
    tag: Optional[str] = None,
):
    """
    High-fidelity tracing for system-wide observability.
    Format: [TAG] [CONTEXT] [ACTION] -> [RESULT]
    """

    # Map tags to colors and icons
    tag_meta = {
        "AGENT": (BLUE, "🤖"),
        "LLM": (YELLOW, "🧠"),
        "REPL": (GREEN, "⚡"),
        "SHEAF": (MAGENTA, "🛡️ "),
        "DREAMER": (CYAN, "💭"),
        "DB": (CYAN, "📊"),
        "RLM": (BLUE, "🔗"),
        "ERROR": (RED, "🚨"),
    }

    color, icon = tag_meta.get(tag, (BOLD, "🔹")) if tag else (BOLD, "")
    tag_str = f"{color}{BOLD}[{tag}] {icon}{RESET} " if tag else ""

    msg = f"{tag_str}{BOLD}[{context}]{RESET} {action}"
    if result is not None:
        # Stringify and truncate long results
        res_str = str(result)
        # Truncation removed by user request for full observability
        msg += f" {BOLD}{color}->{RESET} {res_str}"

    if level == "error":
        logger.error(msg)
    elif level == "warning":
        logger.warning(msg)
    else:
        # Use logger to avoid being captured by REPL stdout callback
        logger.info(msg)

    # Stream to UI if monitor is registered
    if _state.monitor_callback:
        try:
            _state.monitor_callback(msg)
        except (AttributeError, ValueError, TypeError, RuntimeError) as e:
            # Write to stderr to avoid logger recursion, but don't crash
            sys.stderr.write(f"Trace monitor failed: {e}\n")


def banner(title: str):
    """Prints a bright banner to find transitions easily in logs."""
    bar_symbols = "✨" + "=" * 60 + "✨"
    msg = f"\n{BOLD}{CYAN}{bar_symbols}\n {title}\n{bar_symbols}{RESET}\n"
    print(msg)
