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

import os

import tensorflow as tf
tf.enable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = "" #不采用GPU


a = np.random.random((5, 4))
a = a*100
a = a.astype(np.int32)
indices = np.array([True,False,True,False,True])

print(a)
print('*'*10)
#array([[0.47122875, 0.37836802, 0.18210801, 0.341471  ],
#      [0.56551837, 0.27328607, 0.50911876, 0.01179739],
#       [0.75350208, 0.9967817 , 0.94043434, 0.15640884],
#      [0.09511502, 0.96345098, 0.6500849 , 0.04084285],
#       [0.93815553, 0.04821088, 0.10792035, 0.27093746]])
print(a[indices])
#array([[0.47122875, 0.37836802, 0.18210801, 0.341471  ],
#      [0.75350208, 0.9967817 , 0.94043434, 0.15640884],
#       [0.93815553, 0.04821088, 0.10792035, 0.27093746]])
print('*'*10)
c = tf.where(indices)
c=tf.squeeze(c,axis=-1)


print(c)
print('*'*10)

b = tf.gather(tf.constant(a), c)
print(b)
print('*'*10)

#d = a[tf.argsort(tf.cast(a[:, 3]*1,dtype=tf.int32),direction='DESCENDING')] #根据概率降序排序
print(a[:,3])
# d = tf.argsort(a[:,3],direction='DESCENDING')
# bboxes = tf.gather(a, d)
# print(bboxes)

indices = tf.nn.top_k(a[:, 3], k=5).indices
bboxes = tf.gather(a, indices)
print(bboxes)

exit(0)





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