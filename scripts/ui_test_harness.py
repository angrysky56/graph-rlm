import sys
import random
import time
from unittest.mock import MagicMock
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, pyqtSignal, QObject

# Add repo root to path
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- MOCK BACKEND DEPENDENCIES BEFORE IMPORTING UI ---
# This prevents the backend from trying to connect to the database
sys.modules["graph_rlm.backend.src.core.agent"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.agent.core"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.database"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.database.client"] = MagicMock()

# We also need to mock the AgentWorker because the one in UI imports the backend
# However, we are replacing AgentWorker anyway. The issue is that the file *imports* the backend.
# By mocking the backend modules above, the import in AgentWorker will succeed (importing a mock)
# and not trigger the database connection.

from graph_rlm.ui.windows.main_window import MainWindow
import graph_rlm.ui.threads.agent_worker

# ---------------------------------------------------

class MockAgentWorker(QObject):
    # Signals matching AgentWorker
    thoughtCreated = pyqtSignal(dict)
    thoughtUpdated = pyqtSignal(dict)
    linkCreated = pyqtSignal(dict)
    logMessage = pyqtSignal(str, str)
    chatMessage = pyqtSignal(str, str)
    statusChanged = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.node_count = 0
        self.nodes = []

    def start(self):
        self.statusChanged.emit("Mock Agent Ready")
        self.logMessage.emit("INFO", "Mock Agent Started")

        # Simulate some initial chatter
        QTimer.singleShot(1000, lambda: self.chatMessage.emit("user", "Analyze the system architecture."))
        QTimer.singleShot(2000, lambda: self.chatMessage.emit("assistant", "Sure, starting analysis..."))

        # Start generating thoughts
        self.timer = QTimer()
        self.timer.timeout.connect(self._generate_event)
        self.timer.start(800) # Every 800ms

    def stop(self):
        self.timer.stop()
        self.statusChanged.emit("Stopped")

    def wait(self):
        pass

    def send_query(self, prompt):
        self.chatMessage.emit("user", prompt)
        QTimer.singleShot(500, lambda: self.chatMessage.emit("assistant", f"I received: {prompt}. Processing..."))

    def _generate_event(self):
        action = random.choice(["new_node", "update_node", "new_node", "log"])

        if action == "new_node" or not self.nodes:
            self.node_count += 1
            node_id = f"node_{self.node_count}"
            node_data = {
                "id": node_id,
                "status": "pending",
                "label": f"Thinking about step {self.node_count}...",
                "prompt": f"Detailed prompt for step {self.node_count}",
                "priority": random.choice(["high", "medium", "low"]),
                "recency": 1.0
            }
            self.nodes.append(node_id)
            self.thoughtCreated.emit(node_data)
            self.logMessage.emit("INFO", f"Created node {node_id}")

            # Link to a random previous node
            if len(self.nodes) > 1:
                target = node_id
                source = random.choice(self.nodes[:-1])
                self.linkCreated.emit({"source": source, "target": target})

        elif action == "update_node":
            if not self.nodes: return
            node_id = random.choice(self.nodes)
            status = random.choice(["running", "success", "failed", "reflexion"])
            self.thoughtUpdated.emit({
                "id": node_id,
                "status": status,
                "result": "Operation complete." if status == "success" else "Error occurred."
            })
            self.logMessage.emit("DEBUG", f"Updated node {node_id} to {status}")

        elif action == "log":
            self.logMessage.emit("INFO", "System doing background work...")

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Graph RLM (Test Harness)")

    # Apply Stylesheet
    try:
        from pathlib import Path
        style_path = Path(__file__).parent.parent / "graph_rlm" / "ui" / "styles" / "dark.qss"
        if style_path.exists():
            with open(style_path, "r") as f:
                app.setStyleSheet(f.read())
    except Exception as e:
        print(f"Failed to load stylesheet: {e}")

    # Patch the AgentWorker in the main_window module
    import graph_rlm.ui.windows.main_window as mw_module
    mw_module.AgentWorker = MockAgentWorker

    window = mw_module.MainWindow()
    window.show()

    # Run for 5 seconds then exit if running in automation
    if len(sys.argv) > 1 and "--test-run" in sys.argv:
        screenshot_path = None
        if "--screenshot" in sys.argv:
            try:
                idx = sys.argv.index("--screenshot")
                if idx + 1 < len(sys.argv):
                    screenshot_path = sys.argv[idx + 1]
            except ValueError:
                pass

        def close_app():
            if screenshot_path:
                print(f"Taking screenshot to {screenshot_path}")
                # Ensure geometry is laid out
                app.processEvents()
                # Grab window
                pixmap = window.grab()
                pixmap.save(screenshot_path)
            app.quit()

        QTimer.singleShot(3000, close_app)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
