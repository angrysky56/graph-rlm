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

    # Screenshot support for verification
    if "--screenshot" in sys.argv:
        try:
            idx = sys.argv.index("--screenshot")
            if idx + 1 < len(sys.argv):
                path = sys.argv[idx + 1]
                from PyQt6.QtCore import QTimer
                def take_shot():
                    app.processEvents()
                    window.grab().save(path)
                    print(f"Screenshot saved to {path}")
                    app.quit()
                QTimer.singleShot(3000, take_shot)
        except Exception as e:
            print(f"Screenshot failed: {e}")

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
