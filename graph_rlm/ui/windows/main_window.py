from PyQt6.QtWidgets import QMainWindow, QDockWidget, QWidget, QVBoxLayout
from PyQt6.QtCore import Qt

from ..widgets.graph_widget import GraphWidget
from ..widgets.chat_widget import ChatWidget
from ..widgets.log_widget import LogWidget
from ..widgets.inspector_widget import InspectorWidget
from ..threads.agent_worker import AgentWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graph RLM - Recursive Language Model")
        self.resize(1600, 1000)

        # --- Core Components ---
        self.agent_worker = AgentWorker()

        # --- Central Widget: The Graph ---
        self.graph_widget = GraphWidget(self.agent_worker)
        self.setCentralWidget(self.graph_widget)

        # --- Docks ---
        self._create_docks()

        # --- Connecting Signals ---
        self._connect_signals()

        # --- Status Bar ---
        self.statusBar().showMessage("System Ready")

        # Start the Agent Thread
        self.agent_worker.start()

    def _create_docks(self):
        # 1. Chat & Control (Left)
        self.chat_dock = QDockWidget("Chat & Control", self)
        self.chat_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.chat_widget = ChatWidget(self.agent_worker)
        self.chat_dock.setWidget(self.chat_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.chat_dock)

        # 2. Inspector (Right)
        self.inspector_dock = QDockWidget("Node Inspector", self)
        self.inspector_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea)
        self.inspector_widget = InspectorWidget()
        self.inspector_dock.setWidget(self.inspector_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.inspector_dock)

        # 3. Logs (Bottom)
        self.log_dock = QDockWidget("System Logs", self)
        self.log_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.log_widget = LogWidget(self.agent_worker)
        self.log_dock.setWidget(self.log_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)

    def _connect_signals(self):
        # Graph Selection -> Inspector
        self.graph_widget.nodeSelected.connect(self.inspector_widget.display_node)

        # Agent Status -> Status Bar
        self.agent_worker.statusChanged.connect(self.statusBar().showMessage)

        # Agent Errors -> Log Widget (already handled via worker connection in LogWidget)

    def closeEvent(self, event):
        # Clean shutdown
        self.agent_worker.stop()
        self.agent_worker.wait()
        super().closeEvent(event)
