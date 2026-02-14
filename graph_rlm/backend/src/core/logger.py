"""
Structured Logging utility with ANSI color support for Graph-RLM.
"""

import logging
import sys


def ColorFormatterFactory():
    """
    Returns a ColorFormatter instance.
    """

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
            # Don't modify the record levelname globally, just for this format
            orig_levelname = record.levelname
            record.levelname = f"{color}{orig_levelname}{self.RESET}"
            result = super().format(record)
            record.levelname = orig_levelname
            return result

    return ColorFormatter(
        "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Returns a structured logger.
    Only adds a StreamHandler if no parent logger has handlers,
    ensuring we don't get double logs when a root handler exists.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if this logger or ANY of its parents have handlers
    def has_handlers(logger_obj):
        curr = logger_obj
        while curr:
            if curr.handlers:
                return True
            if not curr.propagate:
                break
            curr = curr.parent
        return False

    if not has_handlers(logger):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColorFormatterFactory())
        logger.addHandler(handler)
        # If we added our own handler, we might want to consider propagation
        # but for Graph-RLM we want root capture for UI, so we keep it True.
        # The 'has_handlers' check above prevents us from adding a handler
        # if the root (or any parent) already has one.
        logger.propagate = False

    return logger
