"""
Structured Logging utility with ANSI color support for Graph-RLM.
"""

import logging
import sys


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Returns a structured logger.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        # ANSI Colors
        class ColorFormatter(logging.Formatter):
            """
            Custom formatter to add ANSI colors to log levels.
            """

            LEVEL_COLORS = {
                logging.DEBUG: "\033[90m",  # Grey
                logging.INFO: "\033[94m",  # Blue
                logging.WARNING: "\033[93m",  # Yellow
                logging.ERROR: "\033[91m",  # Red
                logging.CRITICAL: "\033[91m\033[1m",  # Bold Red
            }
            RESET = "\033[0m"

            def format(self, record):
                color = self.LEVEL_COLORS.get(record.levelno, "")
                record.levelname = f"{color}{record.levelname}{self.RESET}"
                return super().format(record)

        formatter = ColorFormatter(
            "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = True

    return logger
