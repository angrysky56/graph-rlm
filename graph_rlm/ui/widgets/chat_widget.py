from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QHBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

class ChatWidget(QWidget):
    def __init__(self, agent_worker):
        super().__init__()
        self.agent_worker = agent_worker

        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(10)

        # Chat History
        self.history = QTextEdit()
        self.history.setReadOnly(True)
        # Remove frame for cleaner look, handled by CSS usually but let's be explicit
        self.history.setFrameShape(QTextEdit.Shape.NoFrame)
        self.layout.addWidget(self.history)

        # Input Area
        input_container = QWidget()
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(5)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter your query...")
        self.input_field.returnPressed.connect(self._send_message)
        # Basic inline styling for input (though stylesheets are better)
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: #1a1a1a;
                color: #e0e0e0;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Segoe UI';
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #00ffcc;
            }
        """)

        self.send_btn = QPushButton("SEND")
        self.send_btn.setFixedWidth(80)
        self.send_btn.clicked.connect(self._send_message)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #0099ff;
            }
            QPushButton:pressed {
                background-color: #005c99;
            }
        """)

        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_btn)
        self.layout.addWidget(input_container)

        # Connect Worker
        self.agent_worker.chatMessage.connect(self.append_message)

    def _send_message(self):
        text = self.input_field.text().strip()
        if not text:
            return

        self.agent_worker.send_query(text)
        self.input_field.clear()

    def append_message(self, role: str, content: str):
        # Bubble Styling
        if role == "assistant":
            # Left aligned, Cyberpunk Cyan accent
            bg_color = "#1e2a33" # Dark blue-grey
            text_color = "#e0e0e0"
            border_color = "#00ffcc"
            align = "left"
            margin_left = "0px"
            margin_right = "40px"
            header = "<span style='color: #00ffcc; font-weight: bold;'>RLM AI</span>"
        else:
            # Right aligned, User accent
            bg_color = "#2d2d2d" # Dark grey
            text_color = "#ffffff"
            border_color = "#666666"
            align = "right"
            margin_left = "40px"
            margin_right = "0px"
            header = "<span style='color: #aaaaaa; font-weight: bold;'>YOU</span>"

        # Formatting content (newlines to breaks)
        formatted_content = content.replace("\n", "<br>")

        html = f"""
        <div style="width: 100%; display: flex; justify-content: {align}; margin-bottom: 10px;">
            <div style="
                background-color: {bg_color};
                color: {text_color};
                border-left: 3px solid {border_color};
                border-radius: 4px;
                padding: 8px 12px;
                margin-left: {margin_left};
                margin-right: {margin_right};
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
                ">
                <div style="margin-bottom: 4px; font-size: 11px;">{header}</div>
                <div>{formatted_content}</div>
            </div>
        </div>
        """

        # We append directly. Note: QTextEdit's HTML support is limited (CSS 2.1 subset).
        # Complex flexbox might not work perfectly. Tables are safer for alignment if flex fails.
        # Let's use a simpler block structure for QTextEdit compatibility.

        if role == "assistant":
            block_html = f"""
            <table width="100%" cellpadding="0" cellspacing="0" border="0">
                <tr>
                    <td align="left">
                        <div style="
                            background-color: {bg_color};
                            color: {text_color};
                            border-left: 3px solid {border_color};
                            padding: 8px;
                            margin-right: 30px;
                        ">
                            <div style="font-size: 10pt; font-weight: bold; color: {border_color}; padding-bottom: 4px;">RLM</div>
                            <div style="font-size: 11pt;">{formatted_content}</div>
                        </div>
                    </td>
                </tr>
            </table>
            <br>
            """
        else:
            block_html = f"""
            <table width="100%" cellpadding="0" cellspacing="0" border="0">
                <tr>
                    <td align="right">
                        <div style="
                            background-color: {bg_color};
                            color: {text_color};
                            border-right: 3px solid {border_color};
                            padding: 8px;
                            margin-left: 30px;
                        ">
                            <div style="font-size: 10pt; font-weight: bold; color: #aaaaaa; padding-bottom: 4px;">YOU</div>
                            <div style="font-size: 11pt;">{formatted_content}</div>
                        </div>
                    </td>
                </tr>
            </table>
            <br>
            """

        self.history.insertHtml(block_html)
        self.history.verticalScrollBar().setValue(self.history.verticalScrollBar().maximum())
