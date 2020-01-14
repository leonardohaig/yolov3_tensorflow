#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:extract_widerperson.py
#       提取wider person 数据集
#Date:20-1-13
#Author:liheng
#Version:V1.0
#============================#


import os
import numpy as np
import scipy.io as sio
import shutil
import cv2


classes = {'1': 'pedestrians',
           '2': 'riders',
           '3': 'partially',
           '4': 'ignore',
           '5': 'crowd'}


def main():
    datasetRootDIr = '/home/liheng/liheng/WiderPerson'
    datasetRootDIr = '/home/liheng/Downloads/WiderPerson'
    imagesDir = os.path.join(datasetRootDIr,'Images')
    annotDir = os.path.join(datasetRootDIr,'Annotations')
    trainfile = os.path.join(datasetRootDIr,'train.txt')
    valfile = os.path.join(datasetRootDIr,'val.txt')


    out_trainPath = './widerPersontrain1.txt'
    out_testPath = './widerPersonTest.txt'

    if 1:#训练集
        in_file = trainfile
        out_file = out_trainPath
    else:
        in_file = valfile
        out_file = out_testPath

    with open(in_file, 'r') as f:
        imgIds = [x for x in f.read().splitlines()]

    with open(out_file,'w') as wf:
        for imgId in imgIds:
            filename = imgId + '.jpg'
            img_path = os.path.join(imagesDir, filename)
            print('Img :%s' % img_path)
            img = cv2.imread(img_path)
            width = img.shape[1]  # 获取图片尺寸
            height = img.shape[0]  # 获取图片尺寸 360

            out_file_line = ' ' #输出文件的一行

            label_path = os.path.join(annotDir, filename) + '.txt'
            with open(label_path) as file:
                line = file.readline()
                count = int(line.split('\n')[0])  # 里面行人个数

                for i in range(count):
                    line = file.readline()
                    cls_id = line.split(' ')[0]
                    if int(cls_id) > 3:
                        continue #不需要类别4 5
                    xmin = int(line.split(' ')[1]) + 1
                    ymin = int(line.split(' ')[2]) + 1
                    xmax = int(line.split(' ')[3]) + 1
                    ymax = int(line.split(' ')[4].split('\n')[0]) + 1

                    _str = '{},{},{},{},{} '.format(xmin, ymin, xmax, ymax, 1)
                    out_file_line += _str
            #保存文件
            if len(out_file_line.strip()):
                out_file_line = '{} {}\n'.format(img_path,out_file_line)
                wf.write(out_file_line)


def mix_up():
    in_trainTxt = ['bdd100k_train.txt','widerPersontrain.txt']
    in_textTxt = ['bdd100k_val.txt','widerPersonTest.txt']
    out_trainTxt = './yoloTrain.txt'
    out_testTxt = './yoloTest.txt'

    if 0:
        in_text = in_trainTxt
        out_text = out_trainTxt
    else:
        in_text = in_textTxt
        out_text = out_testTxt

    with open(out_text,'w') as wf:
        for inTxt in in_text:
            with open(inTxt,'r') as rf:
                for line in rf.readlines():
                    wf.write(line)






if __name__ == '__main__':
    main()
    # mix_up()
    print('Hello world !')