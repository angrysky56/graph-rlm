import logging
import os
import sys

# Ensure we can import core
sys.path.insert(0, os.path.join(os.getcwd(), "graph_rlm/backend"))

from graph_rlm.backend.src.core.log_stream import setup_log_streaming
from graph_rlm.backend.src.core.logger import get_logger
from graph_rlm.backend.src.core.logging import setup_logging


def inspect_loggers():
    print("--- Logging Tree Inspection ---")

    # Simulate full startup
    setup_log_streaming()
    setup_logging(env="development")

    # Trigger a logger creation
    test_logger = get_logger("graph_rlm.core.guardrails")

    curr = test_logger
    while curr:
        print(
            f"Logger: {curr.name or 'root'} (Level: {curr.level}, Propagate: {curr.propagate})"
        )
        for h in curr.handlers:
            fmt = (
                h.formatter._fmt
                if h.formatter and hasattr(h.formatter, "_fmt")
                else "Structlog/Unknown"
            )
            print(f"  Handler: {type(h).__name__} (Level: {h.level}, Formatter: {fmt})")
        curr = curr.parent


if __name__ == "__main__":
    inspect_loggers()
