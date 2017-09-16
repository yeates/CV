# -*- coding:utf8 -*-
# -*- coder：yeates -*-
# -*- 神经网络-手写数字识别 -*-
import sys
reload(sys)
from pyocr import pyocr
from PIL import Image
from pylab import *
import cv2
sys.setdefaultencoding('utf-8')

tools = pyocr.get_available_tools()[:]
print tools[0].image_to_string(Image.open('./temp/temp.png'), lang='eng').encode('GBK', 'ignore')
