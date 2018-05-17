from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np
import os
from math import *
import math
import sys
from sympy import *



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
    return dst,M
  
def rotRandrom(img,factor,size):
    """
    添加透视畸变
    """
    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)], [r(factor), shape[0] - r(factor)], [shape[1] - r(factor),  r(factor)],
                       [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    #print(M)
    #print(M.shape)
    dst = cv2.warpPerspective(img, M, size)
    return dst,M
def get_position(x, y, M):
   """
    x=((M22-M32yo)(M33xo-M13)-(M12-M32xo)(M33yo-M23))/((M22-M32yo)(M11-M31xo)-(M12-M32xo)(M21-M31yo))
    y=((M21-M31yo)(M33xo-M13)-(M11-M31xo)(M33yo-M23))/((M21-M31yo)(M12-M32xo)-(M11-M31xo)(M22-M32yo))
    # 原始图片x，y得到畸变后的坐标
   """
   xo = Symbol('xo')
   yo = Symbol('yo')
   #print(solve([yo+xo-1,3*xo+2*yo-5],[xo,yo]))
   x1 = x - ((M[1][1] - M[2][1]*yo)*(M[2][2]*xo - M[0][2]) - (M[0][1] - M[2][1]*xo)*(M[2][2]*yo - M[1][2]))/((M[1][1] - M[2][1]*yo)*(M[0][0] - M[2][0]*xo) - (M[0][1] - M[2][1]*xo)*(M[1][0] - M[2][0]*yo))
  
   y1 = y - ((M[1][0] - M[2][0]*yo)*(M[2][2]*xo - M[0][2]) - (M[0][0] - M[2][0]*xo)*(M[2][2]*yo - M[1][2]))/((M[1][0] - M[2][0]*yo)*(M[0][1] - M[2][1]*xo) - (M[0][0] - M[2][0]*xo)*(M[1][1] - M[2][1]*yo))
   result = solve([x1,y1],[xo,yo])
   return int(result[xo]),int(result[yo])
    
#添加放射畸变
img = cv2.imread('./data/00002.bmp')
#img_1 = rot(img,r(60)-30,img.shape,30)
img_1,M1 = rot(img,r(45)-30,img.shape,30)
print(get_position(48,33,M1))
cv2.imshow('imh_1',img_1)
cv2.imshow('img',img)

#添加透视畸变
img2 = cv2.imread('./data/00002.bmp')
img_2,M1 = rotRandrom(img2, 12, (img2.shape[1],img2.shape[0]))
get_position(0,0,M1)
cv2.imshow('change_img', img_2)
cv2.imshow('img1', img2)
print(M1)
    
