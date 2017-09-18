# -*- coding:utf8 -*-
# -*- coder：yeates -*-
# -*- 神经网络-手写数字识别:使用百度api的方法 -*-

from PIL import Image
import matplotlib.pyplot as plt
from pylab import *
import hashlib
import urllib
import urllib2
import random
import demjson
import base64
import cv2

def baiduReco(wMat, size_):
    # 图片处理，将所有的需要识别的数字转化成一张图片
    oneMat = np.zeros((28, 28 * 6))
    twoMat = np.zeros((28, 28 * 18))
    for i in range(6):
        for j in range(28):
            for k in range(28):
                oneMat[j, k + i * 28] = wMat[i, j, k]
    oneMat = reverseImage(backgroundAdd(oneMat))
    for i in range(18):
        for j in range(28):
            for k in range(28):
                twoMat[j, k + i * 28] = wMat[i + 6, j, k]
    twoMat = reverseImage(backgroundAdd(twoMat))
    [h1, w1] = oneMat.shape
    [h2, w2] = twoMat.shape
    # 扩大图片分辨率
    sourcePoints = np.float32([[0, 0], [w1, 0], [0, h1], [w1, h1]])
    projecPoints = np.float32([[0, 0], [w1 * 3, 0], [0, h1 * 3], [w1 * 3, h1 * 3]])
    PerspectiveMatrix = cv2.getPerspectiveTransform(sourcePoints, projecPoints)
    rOneMat = cv2.warpPerspective(oneMat, PerspectiveMatrix, (w1 * 3, h1 * 3))

    sourcePoints = np.float32([[0, 0], [w2, 0], [0, h2], [w2, h2]])
    projecPoints = np.float32([[0, 0], [w2 * 3, 0], [0, h2 * 3], [w2 * 3, h2 * 3]])
    PerspectiveMatrix = cv2.getPerspectiveTransform(sourcePoints, projecPoints)
    rTwoMat = cv2.warpPerspective(twoMat, PerspectiveMatrix, (w2 * 3, h2 * 3))
    # 写出图像
    cv2.imwrite('temp/oneNumber.png', rOneMat)
    cv2.imwrite('temp/twoNumber.png', rTwoMat)
    plt.figure('OCR recognize img')
    plt.subplot(121)
    plt.imshow(rOneMat, cmap='gray'), plt.title('first row number')
    plt.subplot(122)
    plt.imshow(rTwoMat, cmap='gray'), plt.title('second row number')
    plt.show()

    youdaoAPI(r'temp/oneNumber.png')
    youdaoAPI(r'temp/twoNumber.png')


def youdaoAPI(filepath):
    appKey = '6b0a6a2f89c0516e'
    secretKey = '0iE7XEvp8iFBY5Ya2g5Hg3VjdLioZulG'

    httpClient = None

    try:
        f = open(filepath, 'rb')  # 二进制方式打开图文件
        img = base64.b64encode(f.read())  # 读取文件内容，转换为base64编码
        f.close()

        detectType = '10011'
        imageType = '1'
        langType = 'en'
        salt = random.randint(1, 65536)

        sign = appKey + img + str(salt) + secretKey
        m1 = hashlib.md5()
        m1.update(sign)
        sign = m1.hexdigest()
        data = {'appKey': appKey, 'img': img, 'detectType': detectType, 'imageType': imageType, 'langType': langType,
                'salt': str(salt), 'sign': sign, 'docType': 'json'}
        data = urllib.urlencode(data)
        req = urllib2.Request('http://openapi.youdao.com/ocrapi', data)

        # response是HTTPResponse对象
        response = urllib2.urlopen(req)
        json_data = response.read()
        print json_data
    except Exception, e:
        print e
    finally:
        if httpClient:
            httpClient.close()


def backgroundAdd(InImg):
    h, w = InImg.shape
    returnImg = np.zeros((120 + h, 350 + w))
    for i in range(h):
        for j in range(w):
            returnImg[i + 60, j + 175] = InImg[i, j]
    return returnImg


def reverseImage(InImg):
    [h, w] = InImg.shape
    for i in range(h):
        for j in range(w):
            InImg[i, j] = 255 - InImg[i, j]
    return InImg