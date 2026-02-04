from PyQt6.QtWidgets import QMainWindow, QDockWidget, QWidget, QVBoxLayout, QMenu
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

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

        # --- Menus ---
        self._create_menus()

        # --- Connecting Signals ---
        self._connect_signals()

        # --- Status Bar ---
        self.statusBar().showMessage("System Ready")

        # Start the Agent Thread AFTER all connections are guaranteed
        # We rely on connections made above.
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

    def _create_menus(self):
        menu_bar = self.menuBar()

        # --- File Menu ---
        file_menu = menu_bar.addMenu("&File")

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # --- View Menu ---
        view_menu = menu_bar.addMenu("&View")

        # Toggle Docks
        toggle_chat = self.chat_dock.toggleViewAction()
        toggle_chat.setText("Chat Panel")
        view_menu.addAction(toggle_chat)

        toggle_inspector = self.inspector_dock.toggleViewAction()
        toggle_inspector.setText("Inspector Panel")
        view_menu.addAction(toggle_inspector)

        toggle_logs = self.log_dock.toggleViewAction()
        toggle_logs.setText("Log Panel")
        view_menu.addAction(toggle_logs)

        view_menu.addSeparator()

        # Physics
        self.physics_action = QAction("Enable Physics", self)
        self.physics_action.setCheckable(True)
        self.physics_action.setChecked(True)
        self.physics_action.toggled.connect(self._toggle_physics)
        view_menu.addAction(self.physics_action)

        # Reset Zoom
        reset_zoom_action = QAction("Reset View", self)
        reset_zoom_action.setShortcut("Ctrl+R")
        reset_zoom_action.triggered.connect(self.graph_widget.view.reset_transform)
        view_menu.addAction(reset_zoom_action)

    def _toggle_physics(self, enabled):
        if hasattr(self.graph_widget.scene, "set_physics_enabled"):
            self.graph_widget.scene.set_physics_enabled(enabled)

    def _connect_signals(self):
        # Graph Selection -> Inspector
        self.graph_widget.nodeSelected.connect(self.inspector_widget.display_node)

        # Agent Status -> Status Bar
        self.agent_worker.statusChanged.connect(self.statusBar().showMessage)

        # Load Complete -> Auto Fit
        self.agent_worker.initialLoadComplete.connect(self.graph_widget.on_loading_finished)

        # Agent Errors -> Log Widget (already handled via worker connection in LogWidget)

    def closeEvent(self, event):
        # Clean shutdown
        self.agent_worker.stop()
        self.agent_worker.wait()
        super().closeEvent(event)
