import sys

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QMainWindow, QPushButton


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Super-resolution Damian Łysomirski")

        button = QPushButton("Press Me!")

        self.setFixedSize(QSize(1200, 800))

        # Set the central widget of the Window.
        self.setCentralWidget(button)