#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:darknet.py
#       实现darknet 53层卷积网络
#Date:2019.05.20
#Author:liheng
#Version:V1.0
#Reference:https://github.com/YunYang1994/tensorflow-yolov3/blob/master/core/common.py
#============================#

__author__ = 'liheng'

import core.common as common
import tensorflow as tf

def darknet53(input_data,trainable):
    '''
    53层卷积网络
    :param input_dat:
    :param trainable:
    :return:
    '''

    with tf.variable_scope('darknet'):
        input_data = common.convolutional(input_data,filters_shape=(3,3,3,32),#3X3卷积，3通道，32个卷积核
                                         trainable=trainable,name='conv0')
        input_data = common.convolutional(input_data,filters_shape=(3,3,32,64),#输入数据厚度为32，输出厚度为64
                                          trainable=trainable,name='conv1',downsample=True)#第1次下采样

        for i in range(1):
            input_data = common.residual_block(input_data,64,32,64,trainable=trainable,
                                               name='residual%d'%(i+0))#输出厚度为64

        input_data = common.convolutional(input_data,filters_shape=(3,3,64,128),
                                          trainable=trainable,name='conv4',downsample=True)#第2次下采样

        for i in range(2):
            input_data = common.residual_block(input_data,128,64,128,trainable=trainable,
                                               name='residual%d'%(i+1))#输出厚度为128

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256),
                                          trainable=trainable, name='conv9', downsample=True)#第3次下采样

        for i in range(8):
            input_data = common.residual_block(input_data,256,128,256,trainable=trainable,
                                               name='residual%d'%(i+3))#输出厚度为256

        route_1 = input_data #52X52 52来源：输入数据大小为416X416，至此，经过了3次下采样，每下采样一次，大小缩小一半，故3次下采样后，大小为416/(2^3)=52
        input_data = common.convolutional(input_data,filters_shape=(3,3,256,512),
                                          trainable=trainable,name='conv26',downsample=True)#第4次下采样

        for i in range(8):
            input_data = common.residual_block(input_data,512,256,512,trainable=trainable,
                                               name='residual%d'%(i+11))#输出厚度为512

        route_2 = input_data #26X26
        input_data = common.convolutional(input_data,filters_shape=(3,3,512,1024),
                                          trainable=trainable,name='conv43',downsample=True)#第5次下采样,该次下采样之后，input_data.shape=(?,13,13,1024).?与批次输入的图片数量有关

        for i in range(4):
            input_data = common.residual_block(input_data,1024,512,1024,trainable=trainable,
                                               name='residual%d'%(i+19))
        #input_data:13X13

        return route_1,route_2,input_data

