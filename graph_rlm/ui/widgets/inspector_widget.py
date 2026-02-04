from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit, QFrame, QHBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

class InspectorWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)

        # --- Header Section (ID & Status) ---
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #1a1a1a; border-radius: 6px; padding: 5px;")
        header_layout = QVBoxLayout(header_frame)

        # ID Label
        self.id_label = QLabel("NODE INSPECTOR")
        self.id_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.id_label.setStyleSheet("color: #ffffff;")
        header_layout.addWidget(self.id_label)

        # Status Badge (Layout)
        status_layout = QHBoxLayout()
        status_label_title = QLabel("STATUS:")
        status_label_title.setStyleSheet("color: #888888; font-size: 10px; font-weight: bold;")

        self.status_badge = QLabel("WAITING")
        self.status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_badge.setStyleSheet("""
            background-color: #333;
            color: #aaa;
            border-radius: 4px;
            padding: 2px 8px;
            font-weight: bold;
            font-size: 11px;
        """)

        status_layout.addWidget(status_label_title)
        status_layout.addWidget(self.status_badge)
        status_layout.addStretch()
        header_layout.addLayout(status_layout)

        self.layout.addWidget(header_frame)

        # --- Content Section ---
        self.layout.addWidget(self._create_section_label("PROMPT / CONTENT"))
        self.content_view = self._create_code_view()
        self.layout.addWidget(self.content_view)

        # --- Result Section ---
        self.layout.addWidget(self._create_section_label("EXECUTION RESULT"))
        self.result_view = self._create_code_view()
        self.layout.addWidget(self.result_view)

    def _create_section_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #00ffcc; font-size: 10px; font-weight: bold; letter-spacing: 1px; margin-top: 10px;")
        return lbl

    def _create_code_view(self):
        view = QTextEdit()
        view.setReadOnly(True)
        view.setFont(QFont("Consolas", 10))
        view.setFrameShape(QFrame.Shape.NoFrame)
        view.setStyleSheet("""
            QTextEdit {
                background-color: #0d0d0d;
                color: #d0d0d0;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 5px;
            }
        """)
        return view

    def display_node(self, data: dict):
        # Update ID
        nid = data.get("id", "N/A")
        self.id_label.setText(f"{nid}")

        # Update Status Badge
        status = data.get("status", "pending")
        self._update_status_badge(status)

        # Update Content
        content = data.get('label', '') or data.get('prompt', '')
        self.content_view.setText(content)

        # Update Result
        result = data.get('result', '') or data.get('execution_summary', '')
        self.result_view.setText(result)

    def _update_status_badge(self, status):
        colors = {
            "active": ("#00ffcc", "#00332a"),     # Cyan / Dark Cyan
            "running": ("#00ffcc", "#00332a"),    # Cyan / Dark Cyan
            "success": ("#00ff66", "#003311"),    # Green / Dark Green
            "failed": ("#ff0055", "#330011"),     # Red / Dark Red
            "error": ("#ff0055", "#330011"),      # Red / Dark Red
            "reflexion": ("#ffcc00", "#332200"),  # Gold / Dark Gold
            "pending": ("#888888", "#222222")     # Grey / Dark Grey
        }

        fg, bg = colors.get(status, colors["pending"])
        self.status_badge.setText(status.upper())
        self.status_badge.setStyleSheet(f"""
            background-color: {bg};
            color: {fg};
            border: 1px solid {fg};
            border-radius: 4px;
            padding: 3px 10px;
            font-weight: bold;
            font-size: 11px;
        """)
