import random
import sys
from PySide6 import *
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import dlib
import cv2
from thread import manWorkerThread,enhWorkerThread
 
class starganv2Page(QWidget):
    def __init__(self):
        super().__init__()
        
        self.layout_QV_group = QGroupBox()
        self.layout_QV = QGridLayout()
        self.layout_QV_group.setLayout(self.layout_QV)


        """
            init
        """
        self.imageLabels = {}
        self.imagePath = {}
        self.currentAlgorithm = "StarGANv2"
        self.statusLabel = QLabel("Current status: Finish")


        """
            operator
        """
        self.layout_OP_group = QGroupBox()
        self.layout_OP_group.setStyleSheet("background-color:rgb(54,64,95);border-radius:4px;color:#ffffff")
        self.layout_OP = QGridLayout()

        # Create label and combobox
        self.label_attributes = QLabel("Gender")
        pe = QPalette()
        pe.setColor(QPalette.WindowText,Qt.white)
        #pe.setColor(QPalette.Background,Qt.blue)
        self.label_attributes.setPalette(pe)
        self.comboBox_attributes = QComboBox()
        self.comboBox_attributes.addItems(["male", "famale"])
        # Beautify combobox
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        self.comboBox_attributes.setFont(font)
        self.comboBox_attributes.setStyleSheet("""
            QComboBox {
                background-color: white;
                color: black;
                padding: 5px;
                border-radius: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: black;
            }
        """)
        self.comboBox_attributes.setMinimumWidth(110)
        self.comboBox_attributes.setMinimumHeight(30)
        # Add label and combobox to layout
        self.layout_OP.addWidget(self.label_attributes, 0, 0, 1, 1)
        self.layout_OP.addWidget(self.comboBox_attributes, 1, 0, 1, 1)


         # Create label and combobox
        self.label_device = QLabel("Device")
        pe = QPalette()
        pe.setColor(QPalette.WindowText,Qt.white)
        #pe.setColor(QPalette.Background,Qt.blue)
        self.label_device.setPalette(pe)
        self.comboBox_device = QComboBox()
        self.comboBox_device.addItems(["GPU", "CPU"])
        # Beautify combobox
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        self.comboBox_device.setFont(font)
        self.comboBox_device.setStyleSheet("""
            QComboBox {
                background-color: white;
                color: black;
                padding: 5px;
                border-radius: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: black;
            }
        """)
        self.comboBox_device.setMinimumWidth(110)
        self.comboBox_device.setMinimumHeight(30)
        self.layout_OP.addWidget(self.label_device, 0, 1, 1, 1)
        self.layout_OP.addWidget(self.comboBox_device, 1, 1, 1, 1)


                 # Create label and combobox
        self.label_enhance = QLabel("Enhance Alg.")
        pe = QPalette()
        pe.setColor(QPalette.WindowText,Qt.white)
        #pe.setColor(QPalette.Background,Qt.blue)
        self.label_enhance.setPalette(pe)
        self.comboBox_enhance = QComboBox()
        self.comboBox_enhance.addItems(["SRGAN", "GFPGAN"])
        # Beautify combobox
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        self.comboBox_enhance.setFont(font)
        self.comboBox_enhance.setStyleSheet("""
            QComboBox {
                background-color: white;
                color: black;
                padding: 5px;
                border-radius: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: black;
            }
        """)
        self.comboBox_enhance.setMinimumWidth(110)
        self.comboBox_enhance.setMinimumHeight(30)
        self.layout_OP.addWidget(self.label_enhance, 0, 2, 1, 1)
        self.layout_OP.addWidget(self.comboBox_enhance, 1, 2, 1, 1)


        self.doButton = filterbtn("Generate")
        self.doButton.clicked.connect(self.manipulate_image)
        self.layout_OP.addWidget(self.doButton, 0, 4, 1, 1)

        self.clearButton = filterbtn("Clear")
        self.clearButton.clicked.connect(self.clear_all)
        self.layout_OP.addWidget(self.clearButton, 0, 5, 1, 1)

        self.enhButton = filterbtn("Enhancement")
        self.enhButton.clicked.connect(self.enhance_image)
        self.layout_OP.addWidget(self.enhButton, 1, 4, 1, 1)

        self.saveButton = filterbtn("Save")
        self.saveButton.clicked.connect(self.save_image)
        self.layout_OP.addWidget(self.saveButton, 1, 5, 1, 1)

        self.layout_OP_group.setLayout(self.layout_OP)
        self.layout_QV.addWidget(self.layout_OP_group)


        """
            image
        """
        self.layout_IM_group = QGroupBox()
        self.layout_IM_group.setStyleSheet("background-color:rgb(54,64,95);border-radius:4px;color:#ffffff")
        self.layout_IM = QGridLayout()


        pe2 = QPalette()
        pe2.setColor(QPalette.WindowText,Qt.black)

        self.titles = ["Original image", "Referance image", "Result"]
        self.currentOriimage = QLabel(self.titles[0])
        self.currentOriimage.setPalette(pe2)
        self.currentOriimage.setFont(QFont("New Roman times",15))
        self.currentOriimage.setAlignment(Qt.AlignCenter)
        self.currentOriimage.setFixedSize(210, 210)
        self.currentOriimage.mousePressEvent = lambda event, t=self.titles[0]: self.open_image(t)
        self.currentOriimage.setStyleSheet("background:rgb(169,169,169);padding:0,0,0,0;")
        self.layout_IM.addWidget(self.currentOriimage,0,0,1,1)
        self.imageLabels[self.titles[0]] = self.currentOriimage

        self.currentRefimage = QLabel(self.titles[1])
        self.currentRefimage.setPalette(pe2)
        self.currentRefimage.setFont(QFont("New Roman times",15))
        self.currentRefimage.setAlignment(Qt.AlignCenter)
        self.currentRefimage.setFixedSize(210, 210)
        self.currentRefimage.mousePressEvent = lambda event, t=self.titles[1]: self.open_image(t)
        self.currentRefimage.setStyleSheet("background:rgb(169,169,169);padding:0,0,0,0;")
        self.layout_IM.addWidget(self.currentRefimage,1,0,1,1)
        self.imageLabels[self.titles[1]] = self.currentRefimage

        self.currentResimage = QLabel(self.titles[2])
        self.currentResimage.setPalette(pe2)
        self.currentResimage.setFont(QFont("New Roman times",15))
        self.currentResimage.setAlignment(Qt.AlignCenter)
        self.currentResimage.setFixedSize(430, 430)
        self.currentResimage.setStyleSheet("background:rgb(169,169,169);padding:0,0,0,0;")
        self.layout_IM.addWidget(self.currentResimage,0,1,2,1)
        self.imageLabels[self.titles[2]] = self.currentResimage
        

        self.layout_IM_group.setLayout(self.layout_IM)
        self.layout_QV.addWidget(self.layout_IM_group)



    def get_current_feature_selection(self):
        return self.comboBox_attributes.currentText()
    
    def get_current_device(self):
        return self.comboBox_device.currentText()


    def manipulate_image(self):
        if not self.imageLabels["Original image"].pixmap():
            QMessageBox.warning(self, "Error", "Please select images for the original slots.")
            return
        if not self.imageLabels["Referance image"].pixmap():
            QMessageBox.warning(self, "Error", "Please select images for the Referance slots.")
            return      
            
        self.thread = manWorkerThread(
            self.imagePath["Original image"],
            self.imagePath["Referance image"],
            self.currentAlgorithm,
            self.get_current_device(),
            self.get_current_feature_selection()

        )

        self.thread.finished.connect(self.on_finish)
        self.thread.error.connect(self.on_error)
        self.thread.start()
        self.statusLabel.setText("Current status: Waiting...")
        self.imageLabels["Result"].setText("Please Waiting....")
        self.imagePath["Result"] = "temp/temp.png"

    
    def enhance_image(self):
        if not self.imageLabels["Result"].pixmap():
            QMessageBox.warning(self, "Error", "Please generating the AI result first.")
            return 
            
        self.thread = enhWorkerThread(
            self.imagePath["Result"],
            self.get_current_device()
        )

        self.thread.finished.connect(self.on_finish)
        self.thread.error.connect(self.on_error)
        self.thread.start()
        self.statusLabel.setText("Current status: Waiting...")
        self.imageLabels["Result"].setText("Please Waiting....") 
            

    def on_finish(self, pixmap, result):
        self.imageLabels["Result"].setPixmap(pixmap)
        self.statusLabel.setText("Current status: Finish")  

    def on_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.statusLabel.setText("Current status: Finish") 

    def save_image(self):
        if self.imageLabels["Result"].pixmap():
            filepath, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "JPEG Files (*.jpg);;PNG Files (*.png)")
            if filepath:
                self.imageLabels["Result"].pixmap().save(filepath)
            QMessageBox.information(self,"Info","Saved successfully.")
        else:
            QMessageBox.warning(self, "Warning!", "There is no generated image to save, please perform image operations first.")



    def open_image(self, title):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if filepath:
            filepath = self.process_img(filepath,title)
            self.imagePath[title] = filepath 
            pixmap = QPixmap(filepath).scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.imageLabels[title].setPixmap(pixmap)


    def process_img(self,filepath,title):
        detector = dlib.get_frontal_face_detector()
        image = cv2.imread(filepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) > 0:
            if image.shape[0] == image.shape[1]:
                return filepath
            face = faces[0]
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            for i in range(1,100):
                try:
                    face_region = image[y-int(h/i):y+h+int(h/5+i), x-int(w/(i+10)):x+w+int(w/(i+10))]
                    cv2.imwrite('temp/'+title+'.png', face_region)
                    break
                except:
                    pass

            return 'temp/'+title+'.png'
        else:
            QMessageBox.warning(self, "Warning!", "No face detected.")
            self.clear_all()


    def clear_all(self):
        self.currentOriimage.clear()
        self.currentOriimage.setText(self.titles[0])
        self.currentRefimage.clear()
        self.currentRefimage.setText(self.titles[1])
        self.currentResimage.clear()
        self.currentResimage.setText(self.titles[2])



class filterbtn(QPushButton):
    def __init__(self, arg):
        super().__init__()

        self.setText(arg)
        self.setFixedSize(170, 22)
        self.setStyleSheet(
            "background-color:rgb(86,100,154);color:#ffffff;border-radius:2px;")

    def enterEvent(self, event):
        self.setStyleSheet(
            "background-color:rgba(86,100,154,0.6);color:#ffffff;border-radius:2px;")

    def leaveEvent(self, event):
        self.setStyleSheet(
            "background-color:rgba(86,100,154,1);color:#ffffff;border-radius:2px;")
        
