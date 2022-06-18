#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Sookie
# @File    :随机森林法

import os
import numpy as np
import cv2
import torch
from sklearn import tree
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = np.genfromtxt('./data/Skin_NonSkin.txt', dtype=np.int32)
labels = data[:, 3]
data = data[:, 0:3]
# RGB转hsv
data = np.reshape(data, (data.shape[0], 1, 3))
data = cv2.cvtColor(np.uint8(data), cv2.COLOR_BGR2HSV)
data = np.reshape(data, (data.shape[0], 3))
# 划分数据集
trainData, testData, trainLables, testLables = train_test_split(data, labels, test_size=0.2, random_state=42)
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(trainData, trainLables)
img_path = './skin_tone_val/image/'
img_name = os.listdir(img_path)
num = len(img_name)
for i in range(num):
    img = cv2.imread(img_path + img_name[i])
    # RGB转hsv
    data = np.reshape(img, (img.shape[0] * img.shape[1], 3))
    data = np.reshape(data, (data.shape[0], 1, 3))
    data = cv2.cvtColor(np.uint8(data), cv2.COLOR_BGR2HSV)
    data = np.reshape(data, (data.shape[0], 3))
    # 预测
    preds = clf.predict(data)
    # 遍历每个像素点是不是肤色，是为1，不是为2，最后将1-2转成0-255
    im = np.reshape(preds, (img.shape[0], img.shape[1], 1))
    mask = (-(im - 1) + 1) * 255
    cv2.imwrite('./skin_tone_val/mask1/' + '{}.png'.format(img_name[i][:-4]), mask)
