# -*- coding:utf8 -*-
# -*- coder：yeates -*-
# -*- 神经网络-手写数字识别 -*-

import os
from PIL import Image
import numpy as np

def binaryzation(data):
    row = data.shape[1]
    col = data.shape[2]
    ret = np.empty(row * col)
    for i in range(row):
        for j in range(col):
            ret[i * col + j] = 0
            if(data[0][i][j] > 127):
                ret[i * col + j] = 1
    return ret

def load_data(data_path, split):
    files = os.listdir(data_path)
    file_num = len(files)
    idx = np.random.permutation(file_num)
    selected_file_num = 42000
    selected_files = []
    for i in range(selected_file_num):
        selected_files.append(files[idx[i]])

    img_mat = np.empty((selected_file_num, 1, 28, 28), dtype = "float32")

    data = np.empty((selected_file_num, 28 * 28), dtype = "float32")
    label = np.empty((selected_file_num), dtype = "uint8")

    print "loading data..."
    for i in range(selected_file_num):
        print i,"/",selected_file_num,"\r",
        file_name = selected_files[i]
        file_path = os.path.join(data_path, file_name)
        img_mat[i] = Image.open(file_path)
        data[i] = binaryzation(img_mat[i])
        label[i] = int(file_name.split('.')[0])
    print ""

    div_line = (int)(split * selected_file_num)
    idx = np.random.permutation(selected_file_num)
    train_idx, test_idx = idx[:div_line], idx[div_line:]
    train_data, test_data = data[train_idx], data[test_idx]
    train_label, test_label = label[train_idx], label[test_idx]

    return train_data, train_label, test_data, test_label

def KNN(test_vec, train_data, train_label, k):
    train_data_size = train_data.shape[0]
    dif_mat = np.tile(test_vec, (train_data_size, 1)) - train_data
    sqr_dif_mat = dif_mat ** 2
    sqr_dis = sqr_dif_mat.sum(axis = 1)

    sorted_idx = sqr_dis.argsort()

    class_cnt = {}
    maxx = 0
    best_class = 0
    for i in range(k):
        tmp_class = train_label[sorted_idx[i]]
        tmp_cnt = class_cnt.get(tmp_class, 0) + 1
        class_cnt[tmp_class] = tmp_cnt
        if(tmp_cnt > maxx):
            maxx = tmp_cnt
            best_class = tmp_class
    return best_class


def KNNclf(wMat, size_):
    np.random.seed(123456)
    train_data, train_label, test_data, test_label = load_data("D:/data_sets/mnist_data", 0.7) # 加载数据集
    print "classifing..."
    for i in range(size_):
        best_class = KNN((wMat[i]).reshape((1, 784)), train_data, train_label, 5)
        print "The %dth Number is identified as %s" % (i, best_class)
