# coding=gbk
###################################################################################

from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
import h5py
from PIL import Image
import os
import time
import math
import random
import hashlib


# 将sub_img图片合并成原图
def merge(images):
    ratio = int(imageSize_H / sub_imgsize)
    h, w = images.shape[1], images.shape[2]  # 16 16
    img = np.zeros((h * ratio, w * ratio, c_dim))  # (128,128,3)
    img_list = []
    mark = 0
    for idx, image in enumerate(images):
        i = idx % ratio
        j = (idx - mark) // ratio
        img[j * h:j * h + h, i * w:i * w + w] = image
        if ((idx + 1) % (ratio * ratio) == 0):
            img = img.astype(images.dtype)
            img_list.append(img)
            mark = idx + 1
    return img_list

##################################################################
def Remove_Zero(array):
    for j in range(array.shape[0]):
        for k in range(array.shape[1]):
            if(array[j][k]<0):
                array[j][k]=0
            if(array[j][k]>1):
                array[j][k]=1
    return array

def Normalize(array):
    maxcols = array.max(axis=0)
    mincols = array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t = np.empty((data_rows, data_cols))
    for i in range(data_cols):
        t[:, i] = (array[:, i] - mincols[i]) / (maxcols[i] - mincols[i])
    return t

def PSNR_1D(arry1,arry2):
    Difference_value = arry1-arry2
    MseD = Difference_value*Difference_value
    Sum = MseD.sum()
    MSE = Sum/(arry1.shape[0]*arry1.shape[1])
    PSNR = 10 * math.log((255.0 * 255.0 / (MSE)), 10)
    return PSNR

def PSNR_3D(im,im2):
    height = im.shape[0]
    width = im.shape[1]
    R = im[:, :, 0] - im2[:, :, 0]
    G = im[:, :, 1] - im2[:, :, 1]
    B = im[:, :, 2] - im2[:, :, 2]
    mser = R * R
    mseg = G * G
    mseb = B * B
    SUM = mser.sum() + mseg.sum() + mseb.sum()
    MSE = SUM / (height * width * 3)
    PSNR = 10 * math.log((255.0 * 255.0 / (MSE)), 10)
    return PSNR
########################################################################
