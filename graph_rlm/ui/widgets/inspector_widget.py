from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit, QScrollArea

class InspectorWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)

        self.id_label = QLabel("ID: -")
        self.layout.addWidget(self.id_label)

        self.status_label = QLabel("Status: -")
        self.layout.addWidget(self.status_label)

        self.layout.addWidget(QLabel("Content:"))
        self.content_view = QTextEdit()
        self.content_view.setReadOnly(True)
        self.layout.addWidget(self.content_view)

        self.layout.addWidget(QLabel("Result:"))
        self.result_view = QTextEdit()
        self.result_view.setReadOnly(True)
        self.layout.addWidget(self.result_view)

    def display_node(self, data: dict):
        self.id_label.setText(f"ID: {data.get('id', 'N/A')}")
        self.status_label.setText(f"Status: {data.get('status', 'N/A')}")
        self.content_view.setText(data.get('label', '') or data.get('prompt', ''))
        self.result_view.setText(data.get('result', '') or data.get('execution_summary', ''))
