"""
Log streaming infrastructure for real-time terminal output to frontend.
Captures all backend logs and streams them via WebSocket.
"""

import asyncio
import logging
from collections import deque
from typing import Callable
from weakref import WeakSet


class LogBuffer:
    """
    Thread-safe buffer that captures log messages for streaming.
    Maintains a fixed-size history and broadcasts to connected clients.
    """

    def __init__(self, max_history: int = 500):
        self.buffer: deque = deque(maxlen=max_history)
        self.subscribers: WeakSet = WeakSet()
        self._lock = asyncio.Lock()

    def add_log(self, message: str):
        """Add a log message to the buffer and notify subscribers."""
        self.buffer.append(message)
        # Notify all subscribers asynchronously
        for callback in list(self.subscribers):
            try:
                callback(message)
            except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
                # Subscriber failed - log but don't crash the buffer
                logging.getLogger(__name__).debug("Subscriber callback failed: %s", e)

    def get_history(self) -> list:
        """Get buffered log history."""
        return list(self.buffer)

    def subscribe(self, callback: Callable[[str], None]):
        """Subscribe to new log messages."""
        self.subscribers.add(callback)

    def unsubscribe(self, callback: Callable[[str], None]):
        """Unsubscribe from log messages."""
        self.subscribers.discard(callback)


class StreamingHandler(logging.Handler):
    """
    Custom logging handler that feeds logs to the LogBuffer.
    """

    def __init__(self, buffer: LogBuffer):
        super().__init__()
        self.log_buffer = buffer
        # Plain text format without ANSI colors for streaming
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_buffer.add_log(msg)
        except Exception:  # pylint: disable=broad-except # noqa: BLE001
            self.handleError(record)


# Global log buffer instance
log_buffer = LogBuffer()


def setup_log_streaming():
    """
    Configure the root logger to stream all logs to the buffer.
    Call this once during application startup.
    """
    # Get root logger
    root_logger = logging.getLogger()

    # Add our streaming handler
    stream_handler = StreamingHandler(log_buffer)
    stream_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(stream_handler)

    # Also capture uvicorn logs
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        logger = logging.getLogger(logger_name)
        logger.addHandler(stream_handler)

    return log_buffer
