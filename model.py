
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
from spectral import SpectralNorm
###########################(HiDDeN Agriculture)###############################################
class ConvBNReLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvBNReLU, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv = SpectralNorm(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1))
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Encoder must accept image and message and output encoded image
class Encoder(nn.Module):
    def __init__(self, in_channels, message_length):
        super(Encoder, self).__init__()
        # if grayscale,input dimension batchsize*1*H*W;else,batchsize*3*H*W	note from lua file
        self.conv0 = nn.Conv2d(in_channels, 64, kernel_size=1,stride=1)
        self.conv1 = ConvBNReLU((64 + message_length ), 64)
        self.conv2 = ConvBNReLU((64 + message_length ), 64)
        self.conv3 = ConvBNReLU((64 + message_length ), 64)
        self.conv4 = ConvBNReLU((64 + message_length ), 64)  # batchsize*64*H*W
        self.conv5 = ConvBNReLU((64 + message_length ), 64)
        self.conv6 = nn.Conv2d(64, in_channels, kernel_size=1, stride=1)
        # x: cover image; y: message_volume

    def forward(self, x, y, S_factor):
        z = self.conv0(x)
        z = torch.cat((y, z), dim=1)
        z = self.conv1(z)
        z = torch.cat((y, z), dim=1)
        z = self.conv2(z)
        z = torch.cat((y, z), dim=1)
        z = self.conv3(z)
        z = torch.cat((y, z), dim=1)
        z = self.conv4(z)
        z = torch.cat((y, z), dim=1)
        z = self.conv5(z)
        z = self.conv6(z)
        return z*S_factor + x
# Decoder must take in a image and return a message
class Decoder(nn.Module):
    def __init__(self, in_channels, message_length):  # message_length is default in lua file,the value 30
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.message_length = message_length
        # if grayscale,input dimension batchsize*1*H*W;else,batchsize*3*H*W	note from lua file
        self.conv0 = nn.Conv2d(in_channels, 64, kernel_size=1,stride=1)
        self.conv1 = ConvBNReLU(64, 64)
        self.conv2 = ConvBNReLU(64, 64)
        self.conv3 = ConvBNReLU(64, 64)
        self.conv4 = ConvBNReLU(64, 64)
        self.conv5 = ConvBNReLU(64, 64)
        self.conv6 = ConvBNReLU(64, 64)
        self.conv7 = ConvBNReLU(64, 64)
        self.conv8 = ConvBNReLU(64, message_length)# batchsize*L*H*W
        self.average_pooling = nn.AdaptiveAvgPool2d(
            1)  # nn.SpatialAdaptiveAveragePooling(1, 1)) in lua file; 2D or 3D? batchsize*L*1*1
        self.linear = SpectralNorm(nn.Linear(message_length, message_length))

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.average_pooling(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out


class Adversary(nn.Module):
    def __init__(self, in_channels):  # message_length is default in lua file,the value 30
        super(Adversary, self).__init__()
        self.in_channels = in_channels
        # if grayscale,input dimension batchsize*1*H*W;else,batchsize*3*H*W	note from lua file
        self.conv1 = ConvBNReLU(in_channels, 64)
        self.conv2 = ConvBNReLU(64, 64)
        self.conv3 = ConvBNReLU(64, 64)
        self.average_pooling = nn.AdaptiveAvgPool2d(1)  # nn.SpatialAdaptiveAveragePooling(1, 1)) in lua file; 2D or 3D? batchsize*L*1*1
        self.linear = SpectralNorm(nn.Linear(64, 1))
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        #x = self.sigmoid(x)
        return x
#####################################################################


