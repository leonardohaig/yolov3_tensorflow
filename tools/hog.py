#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:hog.py
#       
#Date:19-11-5
#Author:liheng
#Version:V1.0
#============================#

from skimage.feature import hog
from skimage import io

import cv2
import numpy as np

image_path = 'images/raccoon-1.jpg'
im = io.imread(image_path,as_gray=True)
normalised_blocks, hog_image = hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8), visualize=True)
# io.imshow(hog_image)



im = cv2.imread(image_path,cv2.IMREAD_COLOR)
im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
cv2.imshow('ori',im)

im = (im*(1.0/255.0)).astype(dtype=np.float32)

mix_image = (hog_image + im)
mix_image = mix_image *(1.0/np.max(mix_image))
sub_image = im - mix_image

mix_image = mix_image*255
mix_image = mix_image.astype(dtype=np.uint8)
cv2.imshow('mix_image',mix_image)
cv2.imshow('sub_image',sub_image)

hog_image = hog_image *(1.0/np.max(hog_image))*255
hog_image = hog_image.astype(dtype=np.uint8)
cv2.imshow('hog',hog_image)



cv2.waitKey(0)