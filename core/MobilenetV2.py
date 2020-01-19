#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:MobilenetV2.py
#       
#Date:19-11-1
#Author:liheng
#Version:V1.0
#============================#

import tensorflow as tf
import core.common as common
from evaluator import *

def MobilenetV2(input_data, training):
    with tf.variable_scope('MobilenetV2'):

        #假设输入input_data.shape:(?,416,416,3)
        conv = common.convolutional(input_data, filters_shape=(3, 3, 3, 32),
                                    trainable=training, name='Conv',
                                    downsample=True, activate=True,
                                    bn=True) #第一次下采样,输出shape(?,208,208,32)

        conv = common.inverted_residual(input_data=conv, input_c=32, output_c=16,
                                        training=training, name='expanded_conv', t=1)#shape:(?,208,208,16)

        conv = common.inverted_residual(name='expanded_conv_1', input_data=conv,
                                        input_c=16, output_c=24, downsample=True,training=training)#第二次下采样,shape:(?,104,104,24)
        conv = common.inverted_residual(name='expanded_conv_2', input_data=conv,
                                        input_c=24, output_c=24, training=training)#shape:(?,104,104,24)

        conv = common.inverted_residual(name='expanded_conv_3', input_data=conv,
                                        input_c=24, output_c=32, downsample=True,training=training)#第三次下采样,shape:(?,52,52,32)
        conv = common.inverted_residual(name='expanded_conv_4', input_data=conv,
                                        input_c=32, output_c=32, training=training)#shape:(?,52,52,32)
        feature_map_s = common.inverted_residual(name='expanded_conv_5', input_data=conv,
                                                 input_c=32, output_c=32,
                                                 training=training)#shape:(?,52,52,32)

        conv = common.inverted_residual(name='expanded_conv_6', input_data=feature_map_s, input_c=32, output_c=64,
                                        downsample=True, training=training)#第四次下采样，shape:(?,26,26,64)
        conv = common.inverted_residual(name='expanded_conv_7', input_data=conv,
                                        input_c=64, output_c=64, training=training)#shape:(?,26,26,64)
        conv = common.inverted_residual(name='expanded_conv_8', input_data=conv,
                                        input_c=64, output_c=64, training=training)#shape:(?,26,26,64)
        conv = common.inverted_residual(name='expanded_conv_9', input_data=conv,
                                        input_c=64, output_c=64, training=training)#shape:(?,26,26,64)

        conv = common.inverted_residual(name='expanded_conv_10', input_data=conv,
                                        input_c=64, output_c=96, training=training)#shape:(?,26,26,96)
        conv = common.inverted_residual(name='expanded_conv_11', input_data=conv,
                                        input_c=96, output_c=96, training=training)#shape:(?,26,26,96)
        feature_map_m = common.inverted_residual(name='expanded_conv_12', input_data=conv,
                                                 input_c=96, output_c=96,training=training)#shape:(?,26,26,96)

        conv = common.inverted_residual(name='expanded_conv_13', input_data=feature_map_m,
                                        input_c=96, output_c=160,downsample=True, training=training)#第五次下采样，shape:(?,13,13,160)
        conv = common.inverted_residual(name='expanded_conv_14', input_data=conv,
                                        input_c=160, output_c=160, training=training)#shape:(?,13,13,160)
        conv = common.inverted_residual(name='expanded_conv_15', input_data=conv,
                                        input_c=160, output_c=160, training=training)#shape:(?,13,13,160)

        conv = common.inverted_residual(name='expanded_conv_16', input_data=conv,
                                        input_c=160, output_c=320, training=training)#shape:(?,13,13,320)

        feature_map_l = common.convolutional(name='Conv_1', input_data=conv, filters_shape=(1, 1, 320, 1280),
                                             trainable=training, activate=True, bn=True)#shape:(?,13,13,1280)

        #在darknet中，输出shap为：
        # route_1.shape=(?,52,52,256)，一个特征点代表8*8的图像  检测小目标，最小检测8*8的图像
        # route_2.shape=(?,26,26,512)，一个特征点代表16*16的图像，检测中目标，最小检测16*16的图像
        # route_3.shape=(?,13,13,1024)， 一个特征点代表32*32像素的图像范围，可以用来检测大目标，最小检测32*32的图像。

        #shape:(?,52,52,32)  (?,26,26,96)  shape:(?,13,13,1280)
    return feature_map_s, feature_map_m, feature_map_l


if __name__ == '__main__':
    graph = tf.get_default_graph()
    input_data = tf.placeholder(dtype=tf.float32,shape=[1,320,320,3],name='input_data')
    trainable = tf.placeholder(dtype=tf.bool, shape=[],name='training')
    s,m,l = MobilenetV2(input_data,trainable)

    flops = evaluate_flops(graph)
    params = evaluate_params(graph)
    print('flops:', flops)#1309224192
    print('params:', params)#2223872
