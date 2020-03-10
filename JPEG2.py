from __future__ import print_function
import cv2
import numpy


Y_QM_50 = numpy.mat([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

UV_QM_50 = numpy.mat([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 98, 99, 99],
    [24, 26, 56, 99, 99, 99, 100, 99],
    [47, 66, 99, 99, 99, 99, 99, 98],
    [100, 99, 99, 99, 99, 99, 99, 99],
    [99, 98, 99, 99, 99, 99, 99, 99],
    [99, 99, 100, 99, 99, 99, 99, 99],
    [99, 99, 99, 98, 99, 99, 99, 99]
])
'''
UV_QM_50 = numpy.mat([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])
'''
def Y_QM(q):
    if (q < 50):
        S = 5000/q
    else:
        S = 200-2*q
    return ((S*Y_QM_50+50.)/100.)

def UV_QM(q):
    if (q < 50):
        S = 5000/q
    else:
        S = 200-2*q
    return ((S*UV_QM_50+50.)/100.)

def DCT(img):
    C_temp = numpy.zeros(img.shape)
    dst = numpy.zeros(img.shape)
    m, n = img.shape
    N = n
    C_temp[0, :] = 1 * numpy.sqrt(1 / N)

    for i in range(1, m):
        for j in range(n):
            C_temp[i, j] = numpy.cos(numpy.pi * i * (2 * j + 1) / (2 * N)
                                  ) * numpy.sqrt(2 / N)

    dst = numpy.dot(C_temp, img)
    dst = numpy.dot(dst, numpy.transpose(C_temp))
    return dst

def IDCT(img):
    C_temp = numpy.zeros(img.shape)
    dst = numpy.zeros(img.shape)
    m, n = img.shape
    N = n
    C_temp[0, :] = 1 * numpy.sqrt(1 / N)

    for i in range(1, m):
        for j in range(n):
            C_temp[i, j] = numpy.cos(numpy.pi * i * (2 * j + 1) / (2 * N)
                                     ) * numpy.sqrt(2 / N)

    dst = numpy.dot(C_temp, img)
    dst = numpy.dot(dst, numpy.transpose(C_temp))
    img_recor = numpy.dot(numpy.transpose(C_temp), dst)
    img_recor1 = numpy.dot(img_recor, C_temp)
    return img_recor1

def JPEG_Mask(img,q):

    stride = 8

    img_Y = img[:, :, 0]
    img_U = img[:, :, 1]
    img_V = img[:, :, 2]

    h, w, c = img.shape
    img_Y_JPEG = numpy.zeros((h, w))
    img_U_JPEG = numpy.zeros((h, w))
    img_V_JPEG = numpy.zeros((h, w))
    img_JPEG = numpy.zeros((h, w, c))

    for x in range(h//stride):
        for y in range(w//stride):
            #sub_image_zeros = numpy.zeros((stride,stride),dtype=numpy.float32)
            Y_block = img_Y[x * stride:(x + 1) * stride, y * stride:(y + 1) * stride]
            U_block = img_U[x * stride:(x + 1) * stride, y * stride:(y + 1) * stride]
            V_block = img_V[x * stride:(x + 1) * stride, y * stride:(y + 1) * stride]
            #sub_image = sub_image_zeros + block  # [8x8]
            Y_DCT_block = DCT(Y_block)
            U_DCT_block = DCT(U_block)
            V_DCT_block = DCT(V_block)

            if (q>0 and q<100):

                Y_IDCT_block = cv2.idct(numpy.array(numpy.multiply(numpy.mat(numpy.rint(Y_DCT_block / Y_QM(q))), Y_QM(q))))
                U_IDCT_block = cv2.idct(numpy.array(numpy.multiply(numpy.mat(numpy.rint(U_DCT_block / UV_QM(q))), UV_QM(q))))
                V_IDCT_block = cv2.idct(numpy.array(numpy.multiply(numpy.mat(numpy.rint(V_DCT_block / UV_QM(q))), UV_QM(q))))
            else:
                for i in range(stride):
                    for j in range(stride):
                        if (i > 4 or j > 4):
                            Y_DCT_block[i, j] = 0
                        if (i > 2 or j > 2):
                            U_DCT_block[i, j] = 0
                            V_DCT_block[i, j] = 0
                Y_IDCT_block = cv2.idct(Y_DCT_block)
                U_IDCT_block = cv2.idct(U_DCT_block)
                V_IDCT_block = cv2.idct(V_DCT_block)


            img_Y_JPEG[x * stride:(x + 1) * stride, y * stride:(y + 1) * stride] = img_Y_JPEG[x * stride:(x + 1) * stride, y * stride:(y + 1) * stride]+Y_IDCT_block
            img_U_JPEG[x * stride:(x + 1) * stride, y * stride:(y + 1) * stride] = img_U_JPEG[x * stride:(x + 1) * stride, y * stride:(y + 1) * stride]+U_IDCT_block
            img_V_JPEG[x * stride:(x + 1) * stride, y * stride:(y + 1) * stride] = img_V_JPEG[x * stride:(x + 1) * stride, y * stride:(y + 1) * stride]+V_IDCT_block

    img_JPEG[:, :, 0] = img_Y_JPEG
    img_JPEG[:, :, 1] = img_U_JPEG
    img_JPEG[:, :, 2] = img_V_JPEG

    img_JPEG = numpy.uint8(img_JPEG)
    return img_JPEG

