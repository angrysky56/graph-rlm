import logging
from typing import Any, Optional

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

# Global monitor callback (e.g. for streaming to UI)
_monitor_callback = None


def register_monitor(callback):
    """Registers a callback(msg: str) to receive trace logs in real-time."""
    global _monitor_callback
    _monitor_callback = callback


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
        logger.info(msg)

    # Stream to UI if monitor is registered
    if _monitor_callback:
        try:
            _monitor_callback(msg)
        except Exception as e:
            # Write to stderr to avoid logger recursion, but don't crash
            import sys

            sys.stderr.write(f"Trace monitor failed: {e}\n")


def banner(title: str):
    """Prints a bright banner to find transitions easily in logs."""
    bar = "✨" + "=" * 60 + "✨"
    msg = f"\n{BOLD}{CYAN}{bar}\n {title}\n{bar}{RESET}\n"
    print(msg)
