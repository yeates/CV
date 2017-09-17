#coding:utf-8
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# 阈值,迭代终止规则
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
images = glob.glob('source/6.jpg')
for fname in images:
    # ======找棋盘格角点=====
    # 棋盘格模板规格
    w = 9
    h = 6
    # 世界坐标系中的棋盘格点
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    # 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = []  # 在世界坐标系中的三维点
    imgpoints = []  # 在图像平面的二维点
    img = cv2.imread(fname)
    img2=cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
    print(ret)
    # 如果找到足够点对，将其存储起来
    #plt.figure()
    #plt.imshow(gray,cmap='gray')
    #plt.show()
    #cv2.imshow('findCorners',gray)
    #cv2.waitKey(50)
    if ret == True:
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        #                灰度图像，初始焦点，搜索窗口，死区，迭代条件
        objpoints.append(objp)
        imgpoints.append(corners)
        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (w,h), corners, ret)
        #                     初始图像，内角点，坐标，是否被找到

        # 标定
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        #   内参数矩阵 矩阵畸变 旋转量 位移量                               点的世界坐标，实际图像坐标，尺寸
        print("内参数矩阵(mtx)：\n",mtx,"\n矩阵畸变(dist):\n",dist,"\n旋转量(rvecs):\n",rvecs,"\n位移量(tvecs):\n",tvecs)
        h, w = img2.shape[:2]
        # 畸变校正
        newcameramtx,roi= cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))  # 自由比例参数
        print("\n相机矩阵(newcamerantx)：\n",newcameramtx)
        # 裁剪
        dst=cv2.undistort(img2, mtx, dist, None, newcameramtx)
        plt.figure()
        plt.subplot(131)
        plt.imshow(img), plt.title('biaoding')
        plt.subplot(132)
        plt.imshow(img2), plt.title('source')
        plt.subplot(133)
        plt.imshow(dst), plt.title('correct')
        plt.show()
        cv2.waitKey(1000)
cv2.destroyAllWindows()

