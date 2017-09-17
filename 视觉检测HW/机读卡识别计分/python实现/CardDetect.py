# -*- coding:utf8 -*-
# -*- coder：yeates -*-
# -*- 机读卡识别 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from PIL import Image
import argparse
from os import listdir
import operator
import ImgProcess as IC

# __________MAIN__________
plt.close('all')
## 读取原图
I = cv2.imread('./img/newMoudleTestImag/123.jpg')

correctedSource, correctedImage = IC.ImgCorrect(I)  # 图像校正
NumberImage = IC.ChooseRecg(correctedImage, correctedSource) # 选择题识别
IC.NumberRecg(NumberImage)                         # 数字识别


# NumberMat = NumberMat.reshape((NumberMat.shape[0], NumberMat.shape[2]))



plt.show()
