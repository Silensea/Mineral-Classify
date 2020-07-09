#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:23:52 2020

@author: silence
"""


from PIL import Image
import os.path
import glob
import numpy as np
import cv2

def compute(img, min_percentile, max_percentile):
    """计算分位点，目的是去掉图1的直方图两头的异常情况"""
    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)

    return max_percentile_pixel, min_percentile_pixel


def aug(src):
    if get_lightness(src)>130:
        print("图片亮度足够，不做增强")
    max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)
    
    # 去掉分位值区间之外的值
    src[src>=max_percentile_pixel] = max_percentile_pixel
    src[src<=min_percentile_pixel] = min_percentile_pixel

    # 将分位值区间拉伸到0到255，这里取了255*0.1与255*0.9是因为可能会出现像素值溢出的情况，所以最好不要设置为0到255。
    out = np.zeros(src.shape, src.dtype)
    cv2.normalize(src, out, 255*0.1,255*0.9,cv2.NORM_MINMAX)

    return out

def get_lightness(src):
	# 计算亮度
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:,:,2].mean()
    print("lightness = {:.4f}".format(lightness))
    
    return  lightness

def convertjpg(jpgfile,outdir,width = 256,height = 192):
#    img=Image.open(jpgfile)
    img = cv2.imread(jpgfile)
    try:
        new_img = cv2.resize(img,(width,height))   
#        new_img = aug(new_img)
        cv2.imwrite(os.path.join(outdir,os.path.basename(jpgfile)), new_img)
    except Exception as e:
        print(e)
        
for jpgfile in glob.glob("/media/silence/Silensea/深度学习/Final_Homework/矿物分类/Data_cross/10_ooid/*.jpg"):
    convertjpg(jpgfile,"/media/silence/Silensea/深度学习/Final_Homework/矿物分类/Data_all_256/train/Ooid")
#    print(jpgfile);
    
#img = cv2.imread("/media/silence/Silensea/深度学习/Final_Homework/矿物分类/Data_cross/4_biotite/001(+).jpg")