# coding=gbk

from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
import h5py
import os
import time
import math
import random
import hashlib
import skimage
#import JPEG2
from PIL import Image,ImageFilter



def Resize(batch_encoded_image_tensor, resize_W, resize_H):
    B,C,W,H = batch_encoded_image_tensor.shape
    batch_encoded_image = batch_encoded_image_tensor.cpu().detach().numpy()
    batch_encoded_image =batch_encoded_image.transpose((0, 2, 3, 1))
    for idx in range(batch_encoded_image.shape[0]):
        encoded_image = batch_encoded_image[idx]
        resize_image = cv2.resize(encoded_image, (resize_W, resize_H))  #still H*W*C after resize
        resize_image = cv2.resize(resize_image,(W, H))
        resize_image = torch.from_numpy(resize_image.transpose((2,0,1))).type(torch.FloatTensor).cuda()
        if (idx == 0):
            batch_noise_image = resize_image.unsqueeze(0)
        else:
            batch_noise_image = torch.cat((batch_noise_image, resize_image.unsqueeze(0)), 0) #batch*H*W*C
    batch_noise_image = Variable(batch_noise_image,requires_grad=True).cuda()   #batch*C*H*W
    return batch_noise_image

def Zero_padding_two(tensor):
    (batch_size,channel,H,W)=tensor.shape
    y_zero = torch.Tensor(torch.zeros(batch_size,channel,1,W)).cuda()
    x_zero = torch.Tensor(torch.zeros(batch_size,channel,H+2,1)).cuda()
    tensor_x = torch.cat((y_zero,tensor),dim=2)
    tensor_x = torch.cat((tensor_x,y_zero),dim=2)
    tensor_x = torch.cat((tensor_x,x_zero),dim=3)
    tensor_x = torch.cat((x_zero,tensor_x), dim=3)
    return tensor_x

def Zero_padding_one(tensor):
    (batch_size,channel,H,W)=tensor.shape
    y_zero = torch.Tensor(torch.zeros(batch_size,channel,1,W)).cuda()
    x_zero = torch.Tensor(torch.zeros(batch_size,channel,H+1,1)).cuda()
    tensor_x = torch.cat((y_zero,tensor),dim=2)
    tensor_x = torch.cat((tensor_x,x_zero),dim=3)
    return tensor_x

def Padding(tensor,img_H,img_W):
    (batch_size, channel, H, W) = tensor.shape
    if((img_H-H)%2==0):
        for i in range(int((img_H-H)/2)):
            tensor = Zero_padding_two(tensor)
    else:
        for i in range(int((img_H-H)/2)):
            tensor = Zero_padding_two(tensor)
        tensor = Zero_padding_one(tensor)
    return tensor

def Gaussian_noise(batch_encoded_image_tensor,Standard_deviation):
    batch_encoded_image = batch_encoded_image_tensor.cpu().detach().numpy()*255
    batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
    for idx in range(batch_encoded_image.shape[0]):
        encoded_image = batch_encoded_image[idx]
        noise_image = skimage.util.random_noise(encoded_image, mode= 'gaussian',clip = False, var = (Standard_deviation) ** 2 )
        noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
        if (idx == 0):
            batch_noise_image = noise_image.unsqueeze(0)
        else:
            batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
    batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
    return batch_noise_image/255

def Salt_Pepper(batch_encoded_image_tensor,Amount):
    batch_encoded_image = batch_encoded_image_tensor.cpu().detach().numpy()*255
    batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
    for idx in range(batch_encoded_image.shape[0]):
        encoded_image = batch_encoded_image[idx]
        noise_image = skimage.util.random_noise(encoded_image, mode='s&p', amount = Amount)
        noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
        if (idx == 0):
            batch_noise_image = noise_image.unsqueeze(0)
        else:
            batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
    batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
    return batch_noise_image/255


def Gussian_blur(batch_encoded_image_tensor,size):
    batch_encoded_image = batch_encoded_image_tensor.cpu().detach().numpy() * 255
    batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
    for idx in range(batch_encoded_image.shape[0]):
        encoded_image = batch_encoded_image[idx]
        noise_image = cv2.GaussianBlur(encoded_image,(size,size),0)
        noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
        if (idx == 0):
            batch_noise_image = noise_image.unsqueeze(0)
        else:
            batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
    batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
    return batch_noise_image / 255

def JPEG_Compression(batch_encoded_image_tensor,Q):
    batch_encoded_image = batch_encoded_image_tensor.cpu().detach().numpy() * 255
    batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
    for idx in range(batch_encoded_image.shape[0]):
        encoded_image = batch_encoded_image[idx]
        noise_image = JPEG2.JPEG_Mask(encoded_image,Q)
        noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
        if (idx == 0):
            batch_noise_image = noise_image.unsqueeze(0)
        else:
            batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
    batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
    return batch_noise_image / 255

def DropOut(batch_encoded_image_tensor,batch_coverd_image_tensor,P):
    mask = np.random.choice([0.0, 1.0], batch_encoded_image_tensor.shape[2:], p=[1 - P, P])
    mask_tensor = torch.tensor(mask, device=batch_encoded_image_tensor.device, dtype=torch.float)
    mask_tensor = mask_tensor.expand_as(batch_encoded_image_tensor)
    noised_image = batch_encoded_image_tensor * mask_tensor + batch_coverd_image_tensor * (1 - mask_tensor)
    return noised_image

def random_float(min, max):
    return np.random.rand() * (max - min) + min

def get_random_rectangle_inside(image, p):
    image_height = image.shape[2]
    image_width = image.shape[3]
    remaining_height = int(np.rint(random_float(p, p) * image_height))
    remaining_width = int(np.rint(random_float(p, p) * image_width))
    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height)
    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)
    return height_start, height_start+remaining_height, width_start, width_start+remaining_width

def Crop(batch_encoded_image_tensor,p):
    h_start, h_end, w_start, w_end = get_random_rectangle_inside(batch_encoded_image_tensor, p)
    noised_image = batch_encoded_image_tensor[
                          :,
                          :,
                          h_start: h_end,
                          w_start: w_end].clone()
    return noised_image

def CropOut(batch_encoded_image_tensor,batch_coverd_image_tensor,p):
    cropout_mask = torch.zeros_like(batch_encoded_image_tensor)
    h_start, h_end, w_start, w_end = get_random_rectangle_inside(batch_encoded_image_tensor, p)
    cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1

    noised_image = batch_encoded_image_tensor * cropout_mask + batch_coverd_image_tensor * (1 - cropout_mask)
    return noised_image

