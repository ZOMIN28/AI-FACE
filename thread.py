import manipulation
import enhancement
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QPixmap

class manWorkerThread(QThread):
    finished = Signal(QPixmap, object)
    #finished = Signal(str,str)
    error = Signal(str) 

    def __init__(self, oriPath, refPath, algorithm, device, currentref):
        super().__init__()
        self.oriPath = oriPath
        self.refPath = refPath
        self.algorithm = algorithm
        self.device = device
        self.currentref = currentref

    def run(self):
        try:
            # 这里应该是图像操作的实际代码
            print("start to manipulate....")
            pixmap,result = manipulation.manipulate(self.oriPath, self.algorithm, self.device, 
                                             self.refPath,self.currentref)
            
            if self.algorithm == "SadTalker":
                self.finished.emit(pixmap,result)
            else:
                self.finished.emit(pixmap,result) 
        except Exception as e:
            self.error.emit(str(e)) 


class enhWorkerThread(QThread):
    finished = Signal(QPixmap, object)
    error = Signal(str) 

    def __init__(self, resPath, device):
        super().__init__()
        self.resPath = resPath
        self.device = device

    def run(self):
        try:
            # 这里应该是图像操作的实际代码
            print("start to enhance....")
            pixmap,result = enhancement.Enhancement(self.resPath,self.device)
            self.finished.emit(pixmap,result) 
        except Exception as e:
            self.error.emit(str(e)) 



class manpathWorkerThread(QThread):
    #finished = Signal(QPixmap, object)
    finished = Signal(str,str)
    error = Signal(str) 

    def __init__(self, oriPath, refPath, algorithm, device, currentref):
        super().__init__()
        self.oriPath = oriPath
        self.refPath = refPath
        self.algorithm = algorithm
        self.device = device
        self.currentref = currentref

    def run(self):
        try:
            # 这里应该是图像操作的实际代码
            print("start to manipulate....")
            file_path,result = manipulation.manipulate(self.oriPath, self.algorithm, self.device, 
                                             self.refPath,self.currentref)
            
            self.finished.emit(file_path,result) 
        except Exception as e:
            self.error.emit(str(e)) 