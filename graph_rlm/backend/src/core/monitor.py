import time
import threading
from .sheaf import SheafMonitor
from .logger import get_logger

logger = get_logger("graph_rlm.monitor")

class BackgroundMonitor:
    def __init__(self, interval: int = 10):
        self.interval = interval
        self.monitor = SheafMonitor()
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("Background Monitor started.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Background Monitor stopped.")

    def _run_loop(self):
        while self.running:
            try:
                energies = self.monitor.scan_and_log()
                if energies:
                    logger.info(f"Monitor Scan Complete using Sheaf Theory. Energy Profile: {energies}")
            except Exception as e:
                logger.error(f"Monitor Loop Error: {e}")

            time.sleep(self.interval)

monitor = BackgroundMonitor()
