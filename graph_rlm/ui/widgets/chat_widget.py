from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QHBoxLayout
from PyQt6.QtCore import Qt

class ChatWidget(QWidget):
    def __init__(self, agent_worker):
        super().__init__()
        self.agent_worker = agent_worker

        self.layout = QVBoxLayout(self)

        # Chat History
        self.history = QTextEdit()
        self.history.setReadOnly(True)
        self.layout.addWidget(self.history)

        # Input Area
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter your query...")
        self.input_field.returnPressed.connect(self._send_message)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._send_message)

        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_btn)
        self.layout.addLayout(input_layout)

        # Connect Worker
        self.agent_worker.chatMessage.connect(self.append_message)

    def _send_message(self):
        text = self.input_field.text().strip()
        if not text:
            return

        self.agent_worker.send_query(text)
        self.input_field.clear()

    def append_message(self, role: str, content: str):
        color = "#007acc" if role == "assistant" else "#e0e0e0"
        align = "left" if role == "assistant" else "right"
        prefix = "<b>RLM:</b> " if role == "assistant" else "<b>You:</b> "

        html = f"<div style='color:{color}; text-align:{align}; margin: 5px;'>{prefix}{content.replace(chr(10), '<br>')}</div><br>"
        self.history.append(html)
