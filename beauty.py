#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Sookie
# @File    :
import math
import os

import cv2
import numpy as np
import torch
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_path = './skin_tone_val/image/'
mask1_path = './skin_tone_val/mask1/'
mask2_path = './skin_tone_val/mask2/'
img_name = os.listdir(img_path)
num = len(img_name)
for i in range(num):
    img = cv2.imread(img_path + img_name[i])
    mask1 = cv2.imread(mask1_path + '{}.png'.format(img_name[i][:-4]))
    mask2 = cv2.imread(mask2_path + '{}.png'.format(img_name[i][:-4]))
    image = transforms.ToTensor()(img)
    mask1 = transforms.ToTensor()(mask1)
    mask2 = transforms.ToTensor()(mask2)
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        mask1 = mask1.squeeze(0).to(device)
        mask2 = mask2.squeeze(0).to(device)

    # 肤色美白曲线法
    beta = 3.0
    if beta == 1:
        v11 = image * mask1
        v12 = image * mask2
    else:
        v11 = torch.log(image * mask1 * (beta - 1) + 1) / math.log(beta)
        v12 = torch.log(image * mask2 * (beta - 1) + 1) / math.log(beta)

    # 随机森林
    v21 = image * (1 - mask1)
    image1 = v11 + v21
    image1 = image1.clamp(0, 1)  # 保证在0-1之间
    image1 = image1.squeeze(0).cpu()  # tensor2opencv
    im1 = image1.numpy()
    maxvalue1 = im1.max()  # 0-1 变为0-255，
    im1 = im1 * 255 / maxvalue1
    mat1 = np.uint8(im1)
    mat1 = mat1.transpose(1, 2, 0)
    cv2.imwrite('./skin_tone_val/skin1/' + '{}.png'.format(img_name[i][:-4]), mat1)

    # hsv+ycrcb
    v22 = image * (1 - mask2)
    image2 = v12 + v22
    image2 = image2.clamp(0, 1)
    image2 = image2.squeeze(0).cpu()
    im2 = image2.numpy()
    maxvalue2 = im2.max()
    im2 = im2 * 255 / maxvalue1
    mat2 = np.uint8(im2)
    mat2 = mat2.transpose(1, 2, 0)
    cv2.imwrite('./skin_tone_val/skin2/' + '{}.png'.format(img_name[i][:-4]), mat2)