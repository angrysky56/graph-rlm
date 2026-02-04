from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit
from PyQt6.QtGui import QFont
from PyQt6.QtCore import QTime

class LogWidget(QWidget):
    def __init__(self, agent_worker):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Consolas", 10))
        # Cleaner look
        self.log_view.setFrameShape(QTextEdit.Shape.NoFrame)
        self.layout.addWidget(self.log_view)

        agent_worker.logMessage.connect(self.append_log)

    def append_log(self, level: str, message: str):
        colors = {
            "INFO": "#cccccc",
            "WARNING": "#ffcc00",
            "ERROR": "#ff4444",
            "DEBUG": "#666666"
        }
        color = colors.get(level, "#ffffff")
        timestamp = QTime.currentTime().toString("HH:mm:ss")

        # HTML formatting with timestamp
        html = f"""
        <div style="font-family: Consolas; margin-bottom: 2px;">
            <span style="color: #444444;">[{timestamp}]</span>
            <span style="color: {color}; font-weight: bold;">[{level}]</span>
            <span style="color: #aaaaaa;"> {message}</span>
        </div>
        """
        self.log_view.append(html)
