# -*- coding:utf8 -*-
# -*- coder：yeates -*-
# -*- 机读卡识别 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from PIL import Image
from imutils.perspective import four_point_transform
from imutils import contours
import argparse
import imutils

# 选择题答案数组初始化
Answer=[]
# 卷子的区域参数
width1=2400
height1=2800
xt1=[0,90,220,350,470,610,700,830,940,1070,1200,1300,1430,1540,1660,1790,1890,2015,2240,2270,2400]
yt1=[900,1000,1070,1140,1210,1300,1375,1450,1500,1580,1650,1750,1810,1880,1950,2150]

#卷子model0判题
def judgey0(y):
    if (y / 5 < 1):
        return  y + 1
    elif y / 5 < 2 and y/5>=1:
        return y % 5 + 20 + 1
    else:
        return y % 5 + 40 + 1
def judgex0(x):
    if(x%5==1):
        return 'A'
    elif(x%5==2):
        return 'B'
    elif(x%5==3):
        return 'C'
    elif(x%5==4):
        return 'D'
def judge0(x,y):
    if x/5<1 :
        #print(judgey0(y))
        return (judgey0(y),judgex0(x))
    elif x/5<2 and x/5>=1:
        #print(judgey0(y)+5)
        return (judgey0(y)+5,judgex0(x))
    elif x/5<3 and x/5>=2:
       # print(judgey0(y)+10)
        return (judgey0(y)+10,judgex0(x))
    else:
        #print(judgey0(y)+15)
        return (judgey0(y)+15,judgex0(x))


# __________MAIN__________
plt.close('all')
## 读取并显示原图
I = cv2.imread('./img/newMoudleTestImag/ipadair (10).jpg')
I = cv2.resize(I, (360, 640))    # 图像大小修改，防止分辨率过大使运行时间太长
source = I
I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)      # 转换为灰度图

plt.figure()
plt.subplot(131)
plt.imshow(I, cmap='gray'), plt.title('Source')

## 自适应阈值，图像分割预处理
blurred = cv2.GaussianBlur(I, (3, 3), 0) # 平滑背景
BinaryImage = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,53,2)
edged = cv2.Canny(BinaryImage, 10, 100)

plt.subplot(132)
plt.imshow(BinaryImage, cmap='gray'), plt.title('Binary Image')
plt.subplot(133)
plt.imshow(edged, cmap='gray'), plt.title('Canny')

## 图像校正
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
docCnt = None

# 确保至少有一个轮廓被找到
if len(cnts) > 0:
    # 将轮廓按大小降序排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # 对排序后的轮廓循环处理
    for c in cnts:
        # 获取近似的轮廓
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 如果近似轮廓有四个顶点，那么就认为找到了答题卡
        if len(approx) == 4:
            docCnt = approx
            break
# 对灰度图都进行四点透视变换(注意，这里是对灰度图进行处理，因为透视变换中的一些插值操作不能在二值图上进行)
corrected = four_point_transform(I, docCnt.reshape(4, 2))
source_cor = four_point_transform(source, docCnt.reshape(4, 2))
# 对灰度图应用二值化算法
BinaryImage = cv2.adaptiveThreshold(corrected,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,53,2)

plt.figure(), plt.subplot(121)
plt.imshow(BinaryImage, cmap='gray'), plt.title('Correct Binary Image')

## 选择题识别
#图形转换为标准方块
BinaryImage = cv2.resize(BinaryImage, (width1, height1), cv2.INTER_LANCZOS4)
source_cor = cv2.resize(source_cor, (width1, height1), cv2.INTER_LANCZOS4)
ChQImg = cv2.blur(BinaryImage, (23, 23))
ChQImg = cv2.threshold(ChQImg, 100, 225, cv2.THRESH_BINARY)[1]

NumImg=cv2.blur(BinaryImage,(15,15))
NumImg=cv2.threshold(NumImg, 170, 255, cv2.THRESH_BINARY)[1]

# 实验发现用边缘检测根本不靠谱……
# 确定选择题答题区
# 在二值图像中查找轮廓，然后初始化题目对应的轮廓列表
cnts = cv2.findContours(ChQImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
questionCnts = []
# 对每一个轮廓进行循环处理
for c in cnts:
    # 计算轮廓的边界框，然后利用边界框数据计算宽高比
    (x, y, w, h) = cv2.boundingRect(c)
    questionCnts.append(c)
    if (w > 60 & h > 20)and y>900 and y<2000:
        questionCnts.append(c)
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.drawContours(source_cor, c, -1, (255, 0, 0), 5, lineType=0)
        cv2.circle(source_cor, (cX, cY), 7, (255, 255, 255), 2)
        Answer.append((cX, cY))

#答案选择输出
IDAnswer=[]
for i in Answer:
    for j in range(0,len(xt1)-1):
        if i[0]>xt1[j] and i[0]<xt1[j+1]:
            for k in range(0,len(yt1)-1):
                if i[1]>yt1[k] and i[1]<yt1[k+1]:
                    judge0(j,k)
                    IDAnswer.append(judge0(j,k))

IDAnswer.sort()
print(IDAnswer)
print("There are %d answer has been recognized."%len(IDAnswer))

# 答案图像显示
plt.subplot(122)
plt.imshow(source_cor, cmap='gray'), plt.title('Choose Status')

plt.show()
