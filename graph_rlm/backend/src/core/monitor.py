"""
Background Monitor for Graph-RLM.
Periodically scans the thought graph for drift and consistency issues.
"""

import threading
import time

from .logger import get_logger
from .sheaf import SheafMonitor

logger = get_logger("graph_rlm.monitor")


class BackgroundMonitor:
    """
    Orchestrates the periodic execution of system monitoring tasks.

    Runs the SheafMonitor in a background thread to analyze system energy
    profiles and thought graph consistency without blocking the main event loop.
    """

    def __init__(self, interval: int = 10):
        """
        Initialize the background monitor.

        Args:
            interval: The scan interval in seconds.
        """
        self.interval = interval
        self.monitor = SheafMonitor()
        self.running = False
        self.thread = None

    def start(self):
        """Start the background monitoring loop in a daemon thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("Background Monitor started.")

    def stop(self):
        """Stop the background monitoring loop."""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Background Monitor stopped.")

    def _run_loop(self):
        """Internal loop that executes the scan at the specified interval."""
        while self.running:
            try:
                energies = self.monitor.scan_and_log()
                if energies:
                    logger.info(
                        "Monitor Scan Complete using Sheaf Theory. Energy Profile: %s",
                        energies,
                    )
            except (RuntimeError, AttributeError, ValueError) as e:
                logger.error("Monitor Loop Error: %s", e)

            time.sleep(self.interval)


monitor = BackgroundMonitor()
