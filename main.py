from PyQt5.QtWidgets import QApplication
from test_window import TestWindow
import sys


def test_window():
    app = QApplication([])
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    test_window()