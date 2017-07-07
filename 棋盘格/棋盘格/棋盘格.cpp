// 棋盘格.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;


int main()
{
	Mat img(1024, 1280, CV_8UC3, Scalar(0, 0, 0));
	int interval = 64;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int swi = (i / interval + j / interval) % 2;
			int val = swi ? 0 : 255;
			img.at<Vec3b>(i, j)[0] = val;
			img.at<Vec3b>(i, j)[1] = val;
			img.at<Vec3b>(i, j)[2] = val;
		}
	}
	imshow("Windows", img);	
	cvWaitKey(0);
	return 0;
}

