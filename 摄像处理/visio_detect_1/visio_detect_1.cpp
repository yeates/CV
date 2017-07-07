// visio_detect_1.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main()
{
	VideoCapture cam;	// 摄像头对象
	VideoWriter vwriter;	
	cam.open(0);		// 打开摄像头
	double w = cam.get(CV_CAP_PROP_FRAME_WIDTH);
	double h = cam.get(CV_CAP_PROP_FRAME_HEIGHT);
	vwriter.open("../test.avi", CV_FOURCC('P', 'I', 'M', '1'), 25, Size(w, h));		//CV_FOURCC自己查手册
	if (!cam.isOpened() || !vwriter.isOpened()) {
		cout << "打开失败" << endl;	
		return 0;
	}
	Mat frame;					// 保存摄像头捕捉到的图像
	Mat tmpImg;
	bool bStop = false;			// 变量
	while(!bStop) {
		cam >> frame;			// 将捕捉到的图像保存到frame中
		cvtColor(frame, tmpImg, CV_RGB2GRAY);					// 灰度转换
		GaussianBlur(tmpImg, tmpImg, Size(11, 11), 10);		// 高斯模糊
		//threshold(tmpImg, tmpImg, 100, 255, CV_THRESH_BINARY);
		Canny(tmpImg, tmpImg, 0, 60);							// 边缘检测
		imshow("Gary Camera", tmpImg);							// 显示
		vwriter << frame;
		//imshow("Camera Frame", frame);
		int key = waitKey(40);
		if (key >= 0) {		// 键盘键入键的时候退出摄像头
			bStop = true;		
		}
	}

	vwriter.release();
	if (cam.isOpened())	
		cam.release();
    return 0;
}

