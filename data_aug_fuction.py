from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np
import os
from math import *
import math
import sys
from sympy import *

def AddSmudginess(img, Smu):
    rows = r(Smu.shape[0] - 50)
    cols = r(Smu.shape[1] - 50)
    adder = Smu[rows:rows + 50, cols:cols + 50]
    adder = cv2.resize(adder, (50, 50))
    #adder = cv2.bitwise_not(adder)
    img = cv2.resize(img,(50,50))
    img = cv2.bitwise_not(img)
    img = cv2.bitwise_and(adder, img)
    img = cv2.bitwise_not(img)
    return img

def rot(img,angel,shape,max_angel):
    """
        添加放射畸变
        img 输入图像
        factor 畸变的参数
        size 为图片的目标尺寸
    """
    size_o = [shape[1],shape[0]]
    size = (shape[1]+ int(shape[0]*cos((float(max_angel )/180) * 3.14)),shape[0])
    interval = abs( int( sin((float(angel) /180) * 3.14)* shape[0]))
    pts1 = np.float32([[0,0],[0,size_o[1]],[size_o[0],0],[size_o[0],size_o[1]]])
    if(angel>0):
        pts2 = np.float32([[interval,0],[0,size[1]  ],[size[0],0  ],[size[0]-interval,size_o[1]]])
    else:
        pts2 = np.float32([[0,0],[interval,size[1]  ],[size[0]-interval,0  ],[size[0],size_o[1]]])
    M  = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)
    return dst

def rotRandrom(img,factor,size):
    """
    添加透视畸变
    """
    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)], [r(factor), shape[0] - r(factor)], [shape[1] - r(factor),  r(factor)],
                       [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    print(M)
    print(M.shape)
    dst = cv2.warpPerspective(img, M, size)
    return dst,M

def get_position(xo, yo, M):
    """
     x=((M22-M32yo)(M33xo-M13)-(M12-M32xo)(M33yo-M23))/((M22-M32yo)(M11-M31xo)-(M12-M32xo)(M21-M31yo))
     y=((M21-M31yo)(M33xo-M13)-(M11-M31xo)(M33yo-M23))/((M21-M31yo)(M12-M32xo)-(M11-M31xo)(M22-M32yo))
    """
    x = Symbol('x')
    y = Symbol('y')
    #x = ((M[1][1] - M[2][1]*yo)*(M[2][2]*xo - M[0][2]) - (M[0][1] - M[2][1]*xo)*(M[2][2]*yo - M[1][2]))/((M[1][1] - M[2][1]*yo)*(M[0][0] - M[2][0]*xo) - (M[0][1] - M[2][1]*xo)*(M[1][0] - M[2][0]*yo))
    #y = ((M[1][0] - M[2][0]*yo)*(M[2][2]*xo - M[0][2]) - (M[0][0] - M[2][0]*xo)*(M[2][2]*yo - M[1][2]))/((M[1][0] - M[2][0]*yo)*(M[0][1] - M[2][1]*xo) - (M[0][0] - M[2][0]*xo)*(M[1][1] - M[2][1]*yo))
    print(solve([((M[1][1] - M[2][1]*yo)*(M[2][2]*xo - M[0][2]) - (M[0][1] - M[2][1]*xo)*(M[2][2]*yo - M[1][2]))//((M[1][1] - M[2][1]*yo)*(M[0][0] - M[2][0]*xo) - (M[0][1] - M[2][1]*xo)*(M[1][0] - M[2][0]*yo)),((M[1][0] - M[2][0]*yo)*(M[2][2]*xo - M[0][2]) - (M[0][0] - M[2][0]*xo)*(M[2][2]*yo - M[1][2]))//((M[1][0] - M[2][0]*yo)*(M[0][1] - M[2][1]*xo) - (M[0][0] - M[2][0]*xo)*(M[1][1] - M[2][1]*yo))]),[x,y])


def tfactor(img):
    """
    添加饱和度光照的噪声
    """
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv[:,:,0] = hsv[:,:,0]*(0.8+ np.random.random()*0.2)
    hsv[:,:,1] = hsv[:,:,1]*(0.3+ np.random.random()*0.7)
    hsv[:,:,2] = hsv[:,:,2]*(0.2+ np.random.random()*0.8)

    img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return img

def random_envirment(img,data_set):
    """
    添加自然环境的噪声
    """
    index=r(len(data_set))
    env = cv2.imread(data_set[index])
    env = cv2.resize(env, (img.shape[1], img.shape[0]))
    bak = (img==0)
    bak = bak.astype(np.uint8)*255
    inv = cv2.bitwise_and(bak, env)
    img = cv2.bitwise_or(inv, img)
    return img

def GenCh(f,val):
    """
    生成中文字符
    """
    img=Image.new("RGB", (45,70),(255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 3),val,(0,0,0),font=f)
    img =  img.resize((23,70))
    A = np.array(img)
    return A

def GenCh1(f,val):
    """
    生成英文字符
    """
    img=Image.new("RGB", (23,70),(255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 2),val.decode('utf-8'),(0,0,0),font=f)
    A = np.array(img)
    return A

def AddGauss(img, level):
    """
    添加高斯模糊
    """
    return cv2.blur(img, (level * 2 + 1, level * 2 + 1))

def r(val):
    return int(np.random.random() * val)

def AddNoiseSingleChannel(single):
    """
    添加高斯噪声
    """
    diff = 255-single.max()
    noise = np.random.normal(0,1+r(6),single.shape)
    noise = (noise - noise.min())/(noise.max()-noise.min())
    noise= diff*noise
    noise= noise.astype(np.uint8)
    dst = single + noise
    return dst

def addNoise(img,sdev = 0.5,avg=10):
    img[:,:,0] =  AddNoiseSingleChannel(img[:,:,0])
    img[:,:,1] =  AddNoiseSingleChannel(img[:,:,1])
    img[:,:,2] =  AddNoiseSingleChannel(img[:,:,2])
    return img


def perspectiveTransf(img, angle_decimal,ratio=1):
    # -------------------------------------------------------------------------------
    # copy from generate_bbox/generate_bbox.py
    pos0, pos1, pos2, pos3, pos4, pos5, pos6 = (70, 50), (230, 50), (360, 50), (460, 50), (570, 50), (680, 50), (
    790, 50)
    text_size = 20
    border0, border1 = (20, 20), (900, 240)
    border20, border21 = (30, 30), (890, 230)
    pos0 = (pos0[0] - 0, pos0[1] + 10)
    pos1 = (pos1[0] - 0, pos1[1] + 10)
    pos2 = (pos2[0] - 0, pos2[1] + 10)
    pos3 = (pos3[0] - 0, pos3[1] + 10)
    pos4 = (pos4[0] - 0, pos4[1] + 10)
    pos5 = (pos5[0] - 0, pos5[1] + 10)
    pos6 = (pos6[0] - 0, pos6[1] + 10)
    leftcorner0, rightcorner0 = pos0, (pos0[0] + text_size + 10, pos0[1] + text_size + 20)
    leftcorner1, rightcorner1 = pos1, (pos1[0] + text_size - 30, pos1[1] + text_size + 10)
    leftcorner2, rightcorner2 = pos2, (pos2[0] + text_size - 30, pos2[1] + text_size + 10)
    leftcorner3, rightcorner3 = pos3, (pos3[0] + text_size - 30, pos3[1] + text_size + 10)
    leftcorner4, rightcorner4 = pos4, (pos4[0] + text_size - 30, pos4[1] + text_size + 10)
    leftcorner5, rightcorner5 = pos5, (pos5[0] + text_size - 30, pos5[1] + text_size + 10)
    leftcorner6, rightcorner6 = pos6, (pos6[0] + text_size - 30, pos6[1] + text_size + 10)
    # -------------------------------------------------------------------------------
    # color,thickness = (255,255,0),5
    # img = cv2.rectangle(img,leftcorner0,rightcorner0,color,thickness)
    # pltimg('original',img)

    (h, w) = img.shape[:2]
    theta = angle_decimal * math.pi / 180

    x1, y1 = 0, 0
    x2, y2 = w, 0
    x3, y3 = 0, h
    x4, y4 = w, h

    if angle_decimal > 0:
        x1 = math.floor(0)
        y1 = math.floor(0)
        x2 = math.floor(w * math.cos(theta))
        y2 = math.floor(w * math.sin(theta))
        x3 = math.floor(x1)
        y3 = math.floor(h)
        x4 = math.floor(x2)
        y4 = math.floor(h + w * math.sin(theta))
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

        dx = math.floor(math.cos(theta) * math.cos(theta))
        dy = math.floor(math.sin(theta))

        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(src=img, M=M, dsize=((int(x2), int(y4))),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        leftcorner0, rightcorner0 = (int(math.floor(leftcorner0[0] * math.cos(theta))),
                                     int(math.floor(leftcorner0[1] + leftcorner0[0] * math.sin(theta)))), (
                                    int(math.floor(rightcorner0[0] * math.cos(theta))),
                                    int(math.floor(rightcorner0[1] + rightcorner0[0] * math.sin(theta))))
        leftcorner1, rightcorner1 = (int(math.floor(leftcorner1[0] * math.cos(theta))),
                                     int(math.floor(leftcorner1[1] + leftcorner1[0] * math.sin(theta)))), (
                                    int(math.floor(rightcorner1[0] * math.cos(theta))),
                                    int(math.floor(rightcorner1[1] + rightcorner1[0] * math.sin(theta))))
        leftcorner2, rightcorner2 = (int(math.floor(leftcorner2[0] * math.cos(theta))),
                                     int(math.floor(leftcorner2[1] + leftcorner2[0] * math.sin(theta)))), (
                                    int(math.floor(rightcorner2[0] * math.cos(theta))),
                                    int(math.floor(rightcorner2[1] + rightcorner2[0] * math.sin(theta))))
        leftcorner3, rightcorner3 = (int(math.floor(leftcorner3[0] * math.cos(theta))),
                                     int(math.floor(leftcorner3[1] + leftcorner3[0] * math.sin(theta)))), (
                                    int(math.floor(rightcorner3[0] * math.cos(theta))),
                                    int(math.floor(rightcorner3[1] + rightcorner3[0] * math.sin(theta))))
        leftcorner4, rightcorner4 = (int(math.floor(leftcorner4[0] * math.cos(theta))),
                                     int(math.floor(leftcorner4[1] + leftcorner4[0] * math.sin(theta)))), (
                                    int(math.floor(rightcorner4[0] * math.cos(theta))),
                                    int(math.floor(rightcorner4[1] + rightcorner4[0] * math.sin(theta))))
        leftcorner5, rightcorner5 = (int(math.floor(leftcorner5[0] * math.cos(theta))),
                                     int(math.floor(leftcorner5[1] + leftcorner5[0] * math.sin(theta)))), (
                                    int(math.floor(rightcorner5[0] * math.cos(theta))),
                                    int(math.floor(rightcorner5[1] + rightcorner5[0] * math.sin(theta))))
        leftcorner6, rightcorner6 = (int(math.floor(leftcorner6[0] * math.cos(theta))),
                                     int(math.floor(leftcorner6[1] + leftcorner6[0] * math.sin(theta)))), (
                                    int(math.floor(rightcorner6[0] * math.cos(theta))),
                                    int(math.floor(rightcorner6[1] + rightcorner6[0] * math.sin(theta))))
    else:
        new_w = math.floor(w * math.cos(theta))
        dy2 = math.floor(new_w * math.sin(theta))
        x1 = math.floor(0)
        y1 = 0
        x2 = math.floor(new_w * math.cos(theta))
        y2 = dy2
        x3 = math.floor(x1)
        y3 = math.floor(h)
        x4 = math.floor(x2)
        y4 = math.floor(h + dy2)
        # adjust
        y1 = y1 + abs(dy2)
        y2 = y2 + abs(dy2)
        y3 = y3 + abs(dy2)
        y4 = y4 + abs(dy2)

        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        M = cv2.getPerspectiveTransform(pts1, pts2)

        color, thickness = (255, 255, 0), 5
        dst = cv2.warpPerspective(src=img, M=M, dsize=((int(x2), int(y3))),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        def new_x(x, theta):  # x is original coordiate
            return int(x * math.cos(theta) * math.cos(theta))

        def new_y(y, x, w, theta):  # x is original coordiate
            new_w = math.floor(w * math.cos(theta))
            new_x_max = new_w * math.cos(theta)
            dy = new_x(x, theta) / float(new_x_max + 1e-6) * abs(w * math.cos(theta) * math.sin(theta))
            return int(abs(w * math.cos(theta) * math.sin(theta)) + y - dy)

        leftcorner0 = new_x(leftcorner0[0], theta), new_y(leftcorner0[1], rightcorner0[0], w, theta)
        leftcorner1 = new_x(leftcorner1[0], theta), new_y(leftcorner1[1], rightcorner1[0], w, theta)
        leftcorner2 = new_x(leftcorner2[0], theta), new_y(leftcorner2[1], rightcorner2[0], w, theta)
        leftcorner3 = new_x(leftcorner3[0], theta), new_y(leftcorner3[1], rightcorner3[0], w, theta)
        leftcorner4 = new_x(leftcorner4[0], theta), new_y(leftcorner4[1], rightcorner4[0], w, theta)
        leftcorner5 = new_x(leftcorner5[0], theta), new_y(leftcorner5[1], rightcorner5[0], w, theta)
        leftcorner6 = new_x(leftcorner6[0], theta), new_y(leftcorner6[1], rightcorner6[0], w, theta)

        rightcorner0 = new_x(rightcorner0[0], theta), new_y(rightcorner0[1], leftcorner0[0], w, theta)
        rightcorner1 = new_x(rightcorner1[0], theta), new_y(rightcorner1[1], leftcorner1[0], w, theta)
        rightcorner2 = new_x(rightcorner2[0], theta), new_y(rightcorner2[1], leftcorner2[0], w, theta)
        rightcorner3 = new_x(rightcorner3[0], theta), new_y(rightcorner3[1], leftcorner3[0], w, theta)
        rightcorner4 = new_x(rightcorner4[0], theta), new_y(rightcorner4[1], leftcorner4[0], w, theta)
        rightcorner5 = new_x(rightcorner5[0], theta), new_y(rightcorner5[1], leftcorner5[0], w, theta)
        rightcorner6 = new_x(rightcorner6[0], theta), new_y(rightcorner6[1], leftcorner6[0], w, theta)

    draw_bbox = False
    if draw_bbox:
        color, thickness = (0, 0, 255), 5
        dst = cv2.rectangle(dst, leftcorner0, rightcorner0, color, thickness)
        dst = cv2.rectangle(dst, leftcorner1, rightcorner1, color, thickness)
        dst = cv2.rectangle(dst, leftcorner2, rightcorner2, color, thickness)
        dst = cv2.rectangle(dst, leftcorner3, rightcorner3, color, thickness)
        dst = cv2.rectangle(dst, leftcorner4, rightcorner4, color, thickness)
        dst = cv2.rectangle(dst, leftcorner5, rightcorner5, color, thickness)
        dst = cv2.rectangle(dst, leftcorner6, rightcorner6, color, thickness)
    # pltimg('affine_dst',dst)
    return dst, leftcorner0, rightcorner0, leftcorner1, rightcorner1, leftcorner2, rightcorner2, leftcorner3, rightcorner3, leftcorner4, rightcorner4, leftcorner5, rightcorner5, leftcorner6, rightcorner6

def funi(i):
    xo,yo,M= i[0],i[1],i[2]
    return [
        ((M[1][1] - M[2][1] * yo) * (M[2][2] * xo - M[0][2]) - (M[0][1] - M[2][1] * xo) * (M[2][2] * yo - M[1][2])) / ((M[1][1] - M[2][1] * yo) * (M[0][0] - M[2][0] * xo) - (M[0][1] - M[2][1] * xo) * (M[1][0] - M[2][0] * yo)),
        ((M[1][0] - M[2][0] * yo) * (M[2][2] * xo - M[0][2]) - (M[0][0] - M[2][0] * xo) * (M[2][2] * yo - M[1][2])) / ((M[1][0] - M[2][0] * yo) * (M[0][1] - M[2][1] * xo) - (M[0][0] - M[2][0] * xo) * (M[1][1] - M[2][1] * yo))
    ]
"""
 <xmin>51</xmin>
 <ymin>17</ymin>
 <xmax>70</xmax>
 <ymax>52</ymax>
"""
img = cv2.imread('./00002.bmp')
#添加放射畸变
#img = cv2.imread('./00002.bmp')
#img_1 = rot(img,r(60)-30,img.shape,30)
#cv2.imshow('imh_1',img_1)
#cv2.imshow('img',img)

#添加透视畸变
#img2 = cv2.imread('./00002.bmp')
#img_2,M1 = rotRandrom(img2, 12, (img2.shape[1],img2.shape[0]))
#cv2.imshow('im_1', img_2)
#cv2.imshow('img1', img2)
#print(M1)
#get_position(51,17,M1)

#a = perspectiveTransf(img2,10,1)
#print(a[0][0])
#cv2.imshow('iii',a[0])
#for i in range(len(a)):
#    print(a[i])

#高斯
img1 = AddGauss(img,2)
cv2.imshow('imh_1',img1)

cv2.waitKey(0)
cv2.destroyAllWindows()

