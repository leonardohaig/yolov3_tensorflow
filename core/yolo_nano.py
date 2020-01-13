#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:yolo_nano.py
#       
#Date:20-1-11
#Author:liheng
#Version:V1.0
#============================#

import tensorflow as tf
import core.common as common
import numpy as np
from evaluator import *

def yolo_nano(input_data, trainable):
    with tf.variable_scope('yolo_nano'):
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 3, 12), trainable=trainable, name='conv0')
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 12, 24),
                                          trainable=trainable, name='conv1', downsample=True)#第1次下采样 (?,208,208,24)
        # print(input_data.shape)
        input_data = common.PEP(input_data, 24, 24, 7, name='PEP0', stride=1,trainable=trainable)
        input_data = common.EP(input_data, 24, 70, name='EP0', stride=2,trainable=trainable)#第2次下采样 (?,104,104,70)
        input_data = common.PEP(input_data, 70, 70, 25, name='PEP1', stride=1,trainable=trainable)
        input_data = common.PEP(input_data, 70, 70, 24, name='PEP2', stride=1,trainable=trainable)
        input_data = common.EP(input_data, 70, 150, name='EP1', stride=2,trainable=trainable)
        input_data = common.PEP(input_data, 150, 150, 56, name='PEP3', stride=1,trainable=trainable)#第3次下采样 (?,52,52,150)

        input_data = common.convolutional(input_data, filters_shape=(1, 1, 150, 150), trainable=trainable, name='conv2')
        input_data = common.FCA(input_data, 150, 8)#(?,52,52,150)
        input_data = common.PEP(input_data, 150, 150, 73, name='PEP4', stride=1,trainable=trainable)
        input_data = common.PEP(input_data, 150, 150, 71, name='PEP5', stride=1,trainable=trainable)
        input_data = common.PEP(input_data, 150, 150, 75, name='PEP6', stride=1,trainable=trainable)
        route_1 = input_data#(?,52,52,150)

        input_data = common.EP(input_data, 150, 325, name='EP2', stride=2,trainable=trainable)#第4次下采样 (?,26,26,325)
        input_data = common.PEP(input_data, 325, 325, 132, name='PEP7', stride=1,trainable=trainable)
        input_data = common.PEP(input_data, 325, 325, 124, name='PEP8', stride=1,trainable=trainable)
        input_data = common.PEP(input_data, 325, 325, 141, name='PEP9', stride=1,trainable=trainable)
        input_data = common.PEP(input_data, 325, 325, 140, name='PEP10', stride=1,trainable=trainable)
        input_data = common.PEP(input_data, 325, 325, 137, name='PEP11', stride=1,trainable=trainable)
        input_data = common.PEP(input_data, 325, 325, 135, name='PEP12', stride=1,trainable=trainable)
        input_data = common.PEP(input_data, 325, 325, 133, name='PEP13', stride=1,trainable=trainable)
        input_data = common.PEP(input_data, 325, 325, 140, name='PEP14', stride=1,trainable=trainable)
        route_2 = input_data#(?,26,26,325)

        input_data = common.EP(input_data, 325, 545, name='EP3', stride=2,trainable=trainable)#第5次下采样 (?,13,13,545)
        input_data = common.PEP(input_data, 545, 545, 276, name='PEP15', stride=1,trainable=trainable)
        input_data = common.convolutional(input_data, filters_shape=(1, 1, 545, 230), trainable=trainable, name='conv3')#(?,13,13,230)
        input_data = common.EP(input_data, 230, 489, name='EP4', stride=1,trainable=trainable)#(?,13,13,489)
        input_data = common.PEP(input_data, 489, 469, 213, name='PEP16', stride=1,trainable=trainable)#(?,13,13,469)
        input_data = common.convolutional(input_data, filters_shape=(1, 1, 469, 189), trainable=trainable, name='conv4')#(?,13,13,189)

        #(?, 52, 52, 150) (?,26,26,325) (?,13,13,189)
        return route_1, route_2, input_data

if __name__ == '__main__':
    graph = tf.get_default_graph()
    input_data = tf.placeholder(dtype=tf.float32,
                                shape=[5, 416, 416, 3],
                                name='input_data')
    trainable = tf.placeholder(dtype=tf.bool, shape=[], name='training')


    conv = yolo_nano(input_data,trainable)

    params = evaluate_params(graph)
    exit(0)