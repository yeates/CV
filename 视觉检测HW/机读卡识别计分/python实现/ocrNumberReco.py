# -*- coding:utf8 -*-
# -*- coder：yeates -*-
# -*- 神经网络-手写数字识别 -*-

from pyocr import pyocr
from PIL import Image
from pylab import *
import cv2

def ocrReco(wMat, size_):
    newMat = np.zeros((28, 28 * size_))
    for i in range(size_):
        for j in range(28):
            for k in range(28):
                newMat[j, k + i * 28] = wMat[i, j, k]

    tempImg = Image.fromarray(uint8(newMat))
    tempImg.save('./temp/temp.png')
    tools = pyocr.get_available_tools()[:]
    print tools[0].image_to_string(Image.open('./temp/temp.png'), lang='eng').encode('GBK', 'ignore')
    # for i in range(size_):

    #     # cv2.imshow('123', wMat[i])
    #     # cv2.waitKey(0)
    #     best_class = tools[0].image_to_string(Image.open('./temp/temp.jpg'), lang='eng')
    #     print "The %dth Number is identified as %s" % (i, best_class)