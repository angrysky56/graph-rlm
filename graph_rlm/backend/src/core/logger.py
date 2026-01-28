import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a structured logger.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        # Format: Timestamp - Level - LoggerName - Message
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False

    return logger
