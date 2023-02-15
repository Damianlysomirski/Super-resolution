import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QDialog, QFileDialog
from PyQt5.QtGui import QFont, QPixmap, QIcon
from PyQt5.QtCore import Qt


# Class QTestWindow to customize your application's test window
class TestWindow(QMainWindow):
    def __init__(self):
        super(TestWindow, self).__init__()

        # Add a title
        self.setWindowTitle("Poprawa rozdzielczośći obrazów")

        # Set size of window
        self.setFixedSize(1150,800)
        
        # first label
        self.label_1 = QtWidgets.QLabel(self)
        self.label_1.setText("Wybierz zdjęcie do poprawy rozdzielczośći: ")
        # moving position
        self.label_1.move(50, 50)
        # setting up the border
        #self.label_1.setStyleSheet("border :1px solid black;")
        # resizing label
        self.label_1.resize(400,40)
        # setting font and size
        self.label_1.setFont(QFont('Times', 10))

        # text field
        self.text_field_1 = QtWidgets.QTextEdit(self)
        self.text_field_1.setReadOnly(True)
        self.text_field_1.move(50,100)
        self.text_field_1.resize(300,50)

        # button
        self.button_1 = QtWidgets.QPushButton(self)
        self.button_1.setText("Przeglądaj")
        self.button_1.move(355, 100)
        self.button_1.resize(100,50)
        self.button_1.setFont(QFont('Times', 10))
        self.button_1.clicked.connect(self.browsefiles)

        # picture 1
        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.move(50,200)
        self.label_2.setPixmap(QPixmap(r'.\resources\no_image.png'))  
        self.label_2.resize(500,500)
        self.label_2.setStyleSheet("border :1px solid black;")
        self.label_2.setAlignment(Qt.AlignCenter)

        # label under picture 1
        #TODO

        # label under picture 2
        #TODO

        # picture 2 
        self.label_3 = QtWidgets.QLabel(self)
        self.label_3.move(600,200)
        self.label_3.setPixmap(QPixmap(r'.\resources\no_image.png'))  
        self.label_3.resize(500,500)
        self.label_3.setStyleSheet("border :1px solid black;")
        self.label_3.setAlignment(Qt.AlignCenter)

        # footer
        self.label_4 = QtWidgets.QLabel(self)
        self.label_4.move(920,770)
        self.label_4.resize(400,20)
        self.label_4.setText("Damian Łysomirski Praca Magisterska")

        # warning
        self.label_warning = QtWidgets.QLabel(self)
        self.label_warning.move(450,40)
        self.label_warning.resize(100,40)

        # button test
        self.button_2 = QtWidgets.QPushButton(self)
        self.button_2.setText("Popraw rozdzielczość")
        self.button_2.move(455, 725)
        self.button_2.resize(200,50)
        self.button_2.setFont(QFont('Times', 10))
        self.button_2.clicked.connect(self.test)

        #Tutaj należy dodać wyjątki
        #Nie wybrano zdjęcia
        #Nie wybrano sieci
        #itd. itd.
        #Potem należy dodać że po poprawie rozdzielczosci będzie znika przycisk popraw rozdzielczośći i pojawia się przycisk zapisz

    # Function to browse file that we want improve the resolution
    def browsefiles(self):
        # Directory to resources
        dir = r'C:\Users\lysom\OneDrive\Pulpit\Praca Dyplomowa\Superesolution_DŁ\resources'
        fname = QFileDialog.getOpenFileName(self, 'Open file', dir, 'Images (*.png *.bmp *.jpg)')
        self.text_field_1.setText(fname[0])
        self.label_2.setPixmap(QPixmap(fname[0]))

    def test(self):
        self.label_warning.setText("ERROR")
        self.label_warning.setStyleSheet("color: red;")
        self.label_warning.setFont(QFont('Times', 20))

 
