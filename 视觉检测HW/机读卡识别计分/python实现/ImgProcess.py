# -*- coding:utf8 -*-
# -*- coder：yeates -*-
# -*- 图片处理模块 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
from imutils.perspective import four_point_transform
from coreNumberReco import *
from ocrNumberReco import *


## 卷子的长宽
width1=800
height1=933

# ================图像校正===============
def ImgCorrect(InImg):
    source = InImg
    InImg = cv2.cvtColor(InImg, cv2.COLOR_RGB2GRAY)      # 转换为灰度图
    plt.subplot(131)
    plt.imshow(InImg, cmap='gray'), plt.title('Source Image')
    ## 自适应阈值，图像分割预处理
    BinaryImage = cv2.adaptiveThreshold(InImg,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,25,10)
    edged = cv2.Canny(BinaryImage, 10, 100)  # parm2:minVal, parm3:maxVal（滞后阈值）

    plt.subplot(132)
    plt.imshow(BinaryImage, cmap='gray'), plt.title('Binary Image')
    plt.subplot(133)
    plt.imshow(edged, cmap='gray'), plt.title('Canny')

    ## 图像校正
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # 轮廓检测，parm2:轮廓检索模式 parm3:轮廓近似方法
    cnts = cnts[0] if imutils.is_cv2() else cnts[1] # 根据opencv版本不同返回不同的图像
    docCnt = None

    # 确保至少有一个轮廓被找到
    if len(cnts) > 0:
        # 将轮廓按大小降序排序
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # 对排序后的轮廓循环处理
        for c in cnts:
            # 获取近似的轮廓
            peri = cv2.arcLength(c, True)   # parm2:是否闭合，返回周长（弧长）
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 轮廓近似, parm2:从原始轮廓到近似轮廓的最大距离

            # 如果近似轮廓有四个顶点，那么就认为找到了答题卡
            if len(approx) == 4:
                docCnt = approx
                break
    # 对灰度图都进行四点透视变换(注意，这里是对灰度图进行处理，因为透视变换中的一些插值操作不能在二值图上进行)
    corrected = four_point_transform(InImg, docCnt.reshape(4, 2))
    corrected_sour = four_point_transform(source, docCnt.reshape(4, 2))
    # 对灰度图应用二值化算法
    BinaryImage = cv2.adaptiveThreshold(corrected,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,53,2)
    plt.subplot(144)
    plt.imshow(BinaryImage, cmap='gray'), plt.title('Correct Binary Image')
    plt.show()
    return corrected_sour, BinaryImage


# ===========================选择题识别==========================
def ChooseRecg(InImg, source):
    # 修改图片大小
    BinaryImage = cv2.resize(InImg, (width1, height1), cv2.INTER_LANCZOS4)
    source = cv2.resize(source, (width1, height1), cv2.INTER_LANCZOS4)
    # 均值滤波
    ChQImg = cv2.blur(BinaryImage, (11, 11))
    # 二进制二值化
    ChQImg = cv2.threshold(ChQImg, 100, 225, cv2.THRESH_BINARY)[1]

    NumImg=cv2.blur(BinaryImage,(6,6))
    NumImg=cv2.threshold(NumImg, 170, 255, cv2.THRESH_BINARY)[1]
    plt.figure()
    plt.subplot(121), plt.imshow(ChQImg, cmap='gray'), plt.title('Choose Region')
    plt.subplot(122), plt.imshow(NumImg, cmap='gray'), plt.title('Number Region')

    # 确定选择题答题区
    # 在二值图像中查找轮廓，然后初始化题目对应的轮廓列表
    cnts = cv2.findContours(ChQImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    questionCnts = []
    # 对每一个轮廓进行循环处理
    k = 0
    for c in cnts:
        k += 1
        # 计算轮廓的边界框，然后利用边界框数据计算宽高比
        (x, y, w, h) = cv2.boundingRect(c)
        questionCnts.append(c)
        if (w > 20 and h > 9)and y>295 and y<693:
            questionCnts.append(c)
            M = cv2.moments(c)              # moments:矩
            cX = int(M["m10"] / M["m00"])   # 计算重心的x坐标
            cY = int(M["m01"] / M["m00"])
            cv2.drawContours(source, c, -1, (255, 0, 0), 5, lineType=0)
            cv2.circle(source, (cX, cY), 2, (0, 0, 255), 2)
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
    plt.figure()
    plt.imshow(source, cmap='gray'), plt.title('Choose Status')
    return NumImg

## 选择题答案数组初始化
Answer=[]

## 选择题判断答题题目模块
xt1=[  0,  30,  73, 116, 156, 203, 233, 276, 313, 356, 400, 433, 476,
       513, 553, 596, 630, 671, 746, 756, 800]
yt1=[300, 333, 356, 380, 403, 433, 458, 483, 500, 526, 550, 583, 603,
       626, 650, 716]
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
        return (judgey0(y),judgex0(x))
    elif x/5<2 and x/5>=1:
        return (judgey0(y)+5,judgex0(x))
    elif x/5<3 and x/5>=2:
        return (judgey0(y)+10,judgex0(x))
    else:
        return (judgey0(y)+15,judgex0(x))



# ======================数字识别======================
def NumberRecg(InImg):
    NumberMat = GetNumberMat(InImg, 24) # 24代表有24个数字需要提取
    #KNNclf(NumberMat, 24)       # 使用KNN进行识别
    ocrReco(NumberMat, 24)     # 使用ocr库进行识别


# 提取数字的区域
# 提取数字的分辨率为32*54 = 1728
def GetNumberMat(NumImg_, size_):
    NumImg = NumImg_
    NumberRect = np.array(NumImg)
    [h, w] = NumberRect.shape
    returnMat = np.zeros((size_, 28, 28), int)
    cnt = 0
    ratio = w / 25
    line_width = 0.17 * ratio + 0.55
    row_1_begin_pos = [0, 2.5 * ratio, 2.8 * ratio]
    row_2_begin_pos = [0, 2.5 * ratio, 5.6 * ratio]
    # 计算第一排的数字图像
    row_1 = np.zeros((7, 5), int)
    row_1[1, 1:3] = [row_1_begin_pos[1] + line_width, row_1_begin_pos[2] + line_width]
    row_1[1, 3:5] = [ratio, ratio*1.7]
    for i in range(2, 7):
        row_1[i, 1:3] = [row_1[i-1, 1] + row_1[i-1, 3] + line_width, row_1[i-1, 2]]
        row_1[i, 3:5] = [row_1[i-1, 3], row_1[i-1, 4]]
    for i in range(1, 7):
        top_left = (row_1[i, 1], row_1[i, 2])
        bottom_right = (row_1[i, 1] + row_1[i, 3], row_1[i, 2] + row_1[i, 4])
        tmpMat = np.zeros((int(ratio*1.7), int(ratio)), bool)
        tmpMat = NumberRect[row_1[i, 2]:row_1[i, 2] + row_1[i, 4], row_1[i, 1]:row_1[i, 1] + row_1[i, 3]]

        tmpMat = projectImg(tmpMat) # 投影变换，将54*32图像压缩为28*28的图像
        # 用dfs消除边缘噪声
        [h, w] = tmpMat.shape
        vis = np.zeros((h, w), int)  # 标记二维数组，判断是否遍历
        for i in range(h):
            for j in range(w):
                if vis[i, j] == 0 and tmpMat[i, j] == 0:
                    [tmpMat, vis] = dfs(tmpMat, i, j, vis, False)

        # cv2.imshow('window_1', tmpMat)
        # cv2.waitKey(0)
        returnMat[cnt] = reverseImage(tmpMat)
        cnt += 1
        # cv2.rectangle(NumberRect, top_left, bottom_right, (0, 0, 255), -1)
        # cv2.imshow('window_2',NumberRect)
        # cv2.waitKey(0)

    # 计算第二排的数字图像
    row_2 = np.zeros((19, 5), int)
    row_2[1, 1:3] = [row_2_begin_pos[1]+ line_width, row_2_begin_pos[2] + line_width]
    row_2[1, 3:5] = [ratio, ratio*1.7]
    for i in range(2,19):
        row_2[i, 1:3] = [row_2[i-1, 1] + row_2[i-1, 3] + line_width, row_2[i-1, 2]]
        row_2[i, 3:5] = [row_2[i-1, 3], row_2[i-1, 4]]
    for i in range(1,19):
        top_left = (row_2[i, 1], row_2[i, 2])
        bottom_right = (row_2[i, 1] + row_2[i, 3], row_2[i, 2] + row_2[i, 4])
        tmpMat = np.zeros((int(ratio*1.7), int(ratio)), bool)
        tmpMat = NumberRect[row_2[i, 2]:row_2[i, 2] + row_2[i, 4], row_2[i, 1]:row_2[i, 1] + row_2[i, 3]]

        tmpMat = projectImg(tmpMat)  # 投影变换，将54*32图像压缩为28*28的图像
        # 用dfs消除边缘噪声
        [h, w] = tmpMat.shape
        vis = np.zeros((h, w), int)  # 标记二维数组，判断是否遍历
        for i in range(h):
            for j in range(w):
                if vis[i, j] == 0 and tmpMat[i, j] == 0:
                    [tmpMat, vis] = dfs(tmpMat, i, j, vis, False)

        # cv2.imshow('a', tmpMat)
        # cv2.waitKey(0)
        returnMat[cnt] = reverseImage(tmpMat)
        cnt += 1
        # cv2.rectangle(NumberRect, top_left, bottom_right, (0, 0, 255), -1)
        # cv2.imshow('test',NumberRect)
        # cv2.waitKey(0)
    return returnMat


def dfs(img, tx, ty, vis, flag):
    ca = [[0,1],[0,-1],[1,0],[-1,0]]
    [h, w] = img.shape
    for i in range(4):
        x = tx + ca[i][0]
        y = ty + ca[i][1]
        if(x < 0 or y < 0):
            flag = True
        if(x < 0 or y < 0 or x >= h or y >= w):
            continue
        if vis[x][y] == 255: continue
        if img[x][y] == 255: continue
        vis[x][y] = 255
        [img, vis] = dfs(img, x, y, vis, flag)
        if flag == True:
            img[tx][ty] = 255
    if flag == True:
        img[tx][ty] = 255
    return [img, vis]


def reverseImage(InImg):
    [h, w] = InImg.shape
    for i in range(h):
        for j in range(w):
            InImg[i, j] = 255 - InImg[i, j]
    return InImg


def projectImg(InImg):
    sourcePoints = np.float32([[0, 0], [32, 0], [0, 54], [32, 54]])
    projecPoints = np.float32([[0, 0], [28, 0], [0, 28], [28, 28]])
    PerspectiveMatrix = cv2.getPerspectiveTransform(sourcePoints, projecPoints)
    returnImg = cv2.warpPerspective(InImg, PerspectiveMatrix, (28, 28))
    return returnImg

