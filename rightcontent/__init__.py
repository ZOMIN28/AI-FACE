import random
import sys
from PySide6.QtWidgets import *
from PySide6 import *
from PySide6.QtCore import Slot
from topnav import Topnav

from starganv2Page import starganv2Page
from hisdPage import hisdPage
from simswapPage import simswapPage
from sadtalkerPage import sadtalkerPage

from bottombar import Bottombar



class Rightcontent(QWidget):
    def __init__(self):
        super().__init__()
        
        
        #右侧最大布局
        self.rightcontent_layout = QVBoxLayout()
        
        
        #导入的顶部导航栏
        self.topnav_group = Topnav()
        self.rightcontent_layout.addWidget(self.topnav_group.topnav_group)
        #self.rightcontent_layout.addStretch()
        
        
        #堆叠布局
        self.rightstack_layout = QStackedLayout()
        #对堆叠布局数据
        self.rightcontentdata = [{"title":"StarGANv2"},
                            {"title":"HiSD"},
                            {"title":"SimSwap"},
                            {"title":"SadTalker"}]
        


        self.stacklayout_starganv2Page = starganv2Page()
        
        self.stacklayout_hisdPage = hisdPage()
        
        self.stacklayout_simswapPage = simswapPage()

        self.stacklayout_sadtalkerPage= sadtalkerPage()


        #堆叠布局导入四个tab
        self.rightstack_layout.addWidget(self.stacklayout_starganv2Page.layout_QV_group)
        self.rightstack_layout.addWidget(self.stacklayout_hisdPage.layout_QV_group)
        self.rightstack_layout.addWidget(self.stacklayout_simswapPage.layout_QV_group)
        self.rightstack_layout.addWidget(self.stacklayout_sadtalkerPage.layout_QV_group)

        
        #堆叠布局纺放入大布局里
        self.rightcontent_layout.addLayout(self.rightstack_layout)


        #导入右侧bottombar
        self.bottombar_group = Bottombar()
        self.rightcontent_layout.addWidget(self.bottombar_group.bottombar_group)
        


    
    #接受leftmenu发出的信号方法
    @Slot(str)
    def getmenuindex(self, msg):
        self.rightstack_layout.setCurrentIndex(msg["index"])
        self.rightstack_layout.update