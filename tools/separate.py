#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:separate.py
#       将train.txt中的文件路径+图像名称修改为图像名称
#Date:19-12-31
#Author:liheng
#Version:V1.0
#============================#

import os

def separate(annot_path,new_annot_path):
    with open(annot_path,'r') as old_file, open(new_annot_path,'w') as new_file:
        for line in old_file:
            line = line.rsplit('/',maxsplit=1)
            new_file.write(line[1])


if __name__ == '__main__':
    # annot_path = 'bdd100k_train.txt'
    # new_annot_path = 'new_bdd100k_train.txt'
    annot_path = '/home/liheng/liheng/liheng/yolov3_tensorflow/data/classes/train_yoloTF.txt'
    new_annot_path = '/home/liheng/liheng/liheng/yolov3_tensorflow/data/classes/train_yoloTF1.txt'
    separate(annot_path,new_annot_path)
