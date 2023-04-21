import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QDialog, QFileDialog
from PyQt5.QtGui import QFont, QPixmap, QIcon
from PyQt5.QtCore import Qt

#To delete later
import cv2
from PIL import Image
from PIL.ImageQt import ImageQt

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

        #Tutaj należy dodać wyjątki
        #Nie wybrano zdjęcia
        #Nie wybrano sieci
        #itd. itd.
        #Potem należy dodać że po poprawie rozdzielczosci będzie znika przycisk popraw rozdzielczośći i pojawia się przycisk zapisz

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

        # label5
        self.label_5 = QtWidgets.QLabel(self)
        self.label_5.setText("Wybierz rodzaj sieci: ")
        self.label_5.move(600, 50)
        self.label_5.resize(400,40)
        self.label_5.setFont(QFont('Times', 10))

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

        #Ten fragment też jest potem do usnięcia
        self.combobox = QtWidgets.QComboBox(self)
        self.combobox.move(600, 100)
        self.combobox.resize(400,40)
        self.combobox.addItems(['Bicubic 2x', 'SRCNN 2x', 'SRResNet'])
        self.combobox.setFont(QFont('Times', 10))

    # Function to browse file that we want improve the resolution
    def browsefiles(self):
        # Directory to resources
        dir = r'C:\Users\lysom\OneDrive\Pulpit\Praca Dyplomowa\Superesolution_DŁ\resources'
        fname = QFileDialog.getOpenFileName(self, 'Open file', dir, 'Images (*.png *.bmp *.jpg)')
        self.text_field_1.setText(fname[0])
        self.label_2.setPixmap(QPixmap(fname[0]))

        """
        This fragment to delete later 
        """
        #Tutaj wgl coś crashuje czasami..
        # img = Image.open(fname[0])
        # w, h = img.size
        # resize_image = img.resize((int(w/2), int(h/2)), Image.BICUBIC)  
        # resize_imagee = ImageQt(resize_image)
        # self.label_2.setPixmap(QPixmap.fromImage(resize_imagee))
       

    def test(self):
        #self.label_warning.setText("ERROR")
        #self.label_warning.setStyleSheet("color: red;")
        #self.label_warning.setFont(QFont('Times', 20))
        pass
 
