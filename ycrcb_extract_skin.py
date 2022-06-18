#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Sookie
# @File    :hsv+ ycrcb提取出皮肤区域
import os
import cv2
import numpy as np
import torch


img_path = './skin_tone_val/image/'
img_name = os.listdir(img_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num = len(img_name)
for i in range(num):
    img = cv2.imread(img_path + img_name[i])
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    lower_HSV_values = np.array([7, 28, 50], dtype="uint8")
    upper_HSV_values = np.array([20, 255, 255], dtype="uint8")
    lower_YCbCr_values = np.array((0, 140, 100), dtype="uint8")
    upper_YCbCr_values = np.array((255, 175, 120), dtype="uint8")

    mask_ycrcb = cv2.inRange(ycrcb_img, lower_YCbCr_values, upper_YCbCr_values)
    mask_hsv = cv2.inRange(hsv_img, lower_HSV_values, upper_HSV_values)
    mask = cv2.add(mask_hsv, mask_ycrcb)

    image_foreground = cv2.erode(mask, None, iterations=3)
    dilated_mask = cv2.dilate(mask, None, iterations=3)
    _, image_background = cv2.threshold(dilated_mask, 1, 125, cv2.THRESH_BINARY)
    image_marker = cv2.add(image_foreground, image_background)
    _, image_mask = cv2.threshold(image_marker, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('./skin_tone_val/mask2/' + '{}.png'.format(img_name[i][:-4]), image_mask)
