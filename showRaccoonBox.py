#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:showRaccoonBox.py
#       显示raccoon数据集的groundtruth box，以确认box是否无误
#Date:2019.08.22
#Author:liheng
#Version:V1.0
#============================#

__author__ = 'liheng'

import cv2
from core.config import cfg

strAnnotationPath = cfg.TRAIN.ANNOT_PATH # 标签及box文件路径

with open(strAnnotationPath,'r') as annotFile:
    for line in annotFile: # 读取一行
        content = line.split(' ') # 空格分割，读取文件路径

        imagePath=content[0] # 图像路径
        box_id = content[1].strip().strip('\n').split(',')

        image = cv2.imread(imagePath,cv2.IMREAD_UNCHANGED)
        cv2.rectangle(image,(int(box_id[0]),int(box_id[1])),(int(box_id[2]),int(box_id[3])),(0,255,0),1,lineType=cv2.LINE_AA)
        cv2.imshow("image",image)
        cv2.waitKey(0)



