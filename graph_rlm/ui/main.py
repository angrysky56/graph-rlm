import sys
import signal
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon

from .windows.main_window import MainWindow

def main():
    # Allow Ctrl+C to kill the app from terminal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    app.setApplicationName("Graph RLM")

    # Apply Stylesheet
    try:
        from pathlib import Path
        style_path = Path(__file__).parent / "styles" / "dark.qss"
        if style_path.exists():
            with open(style_path, "r") as f:
                app.setStyleSheet(f.read())
    except Exception as e:
        print(f"Failed to load stylesheet: {e}")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
