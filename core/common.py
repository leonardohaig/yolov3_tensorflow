#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:common.py
#       定义yolo v3中的卷积层、残差层以及张量拼接和上采样函数
#Date:2019.05.20
#Author:liheng
#Version:V1.0
#Reference:https://github.com/YunYang1994/tensorflow-yolov3/blob/master/core/common.py
#============================#

__author__ = 'liheng'

import tensorflow as tf

def convolutional(input_data, filters_shape,
                  trainable,name,
                  downsample=False,activate=True,bn=True):
    '''
    yolo v3 中的 卷积层=卷积+BN+LeakyReLU
    :param input_data:
    :param filters_shape:
    :param trainable:
    :param name:
    :param downsample:
    :param activate:
    :param bn:
    :return:
    '''
    with tf.variable_scope(name):
        if downsample:
            pad_h,pad_w = (filters_shape[0]-2)//2+1,(filters_shape[1]-2)//2+1#填充高，宽
            paddings = tf.constant([[0,0],[pad_h,pad_h],[pad_w,pad_w],[0,0]])
            input_data = tf.pad(input_data,paddings,'CONSTANT')
            strides = (1,2,2,1)
            padding = 'VALID'
        else:
            strides = (1,1,1,1)
            padding = 'SAME'

        weight = tf.get_variable(name='weight',dtype=tf.float32,trainable=True,
                                 shape=filters_shape,initializer=tf.random_normal_initializer(stddev=0.01))

        #conv2d参数说明：
        #第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，
        #   具有[batch, in_height, in_width, in_channels]这样的shape，
        #   具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数(or 厚度)]，注意这是一个4维的Tensor，
        #   要求类型为float32和float64其中之一
        #第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，
        #   具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
        #   具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，
        #   有一个地方需要注意，第三维in_channels，就是参数input的第四维
        #第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4。
        #   对于图片，因为只有两维，通常strides取[1，stride，stride，1]
        #第四个参数padding：string类型的量，只能是"SAME", "VALID"其中之一，这个值决定了不同的卷积方式
        #第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
        #结果返回一个Tensor，这个输出，就是我们常说的feature map，shape仍然是[batch, height, width, channels]这种形式。
        conv = tf.nn.conv2d(input=input_data,filter=weight,strides=strides,padding=padding)

        if bn:
            #批规范化
            #原理公式：y=γ(x-μ)/σ+β.其中x是输入，y是输出，μ是均值，σ是方差，γ和β是缩放（scale）、偏移（offset）系数
            #       μ和σ，在训练的时候，使用的是batch内的统计值，测试/预测的时候，采用的是训练时计算出的滑动平均值
            #总的来说就是对于同一batch的input，假设输入大小为[batch_num, height, width, channel]，
            #   逐channel地计算同一batch中所有数据的mean和variance，
            #   再对input使用mean和variance进行归一化，最后的输出再进行线性平移，得到batch_norm的最终结果。
            #伪代码如下(https://blog.csdn.net/huitailangyz/article/details/85015611)：
            '''
            for i in range(channel):
                x = input[:, :, :, i]
                mean = mean(x)
                variance = variance(x)
                x = (x - mean) / sqrt(variance)
                x = scale * x + offset
                input[:, :, :, i] = x

            '''
            conv = tf.layers.batch_normalization(conv,beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(),
                                                 training=trainable)#BN操作，训练时training=True，测试时training=False
        else:
            bias = tf.get_variable(name='bias',shape=filters_shape[-1],trainable=True,
                                   dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv,bias)

        if activate:
            conv = tf.nn.leaky_relu(conv,alpha=0.1)#激活函数

    return conv

def residual_block(input_data,input_channel,filter_num1,filter_num2,trainable,name):
    '''
    残差模块，残差输出 = 输入 + (1X1convolutional输入 + 3X3convolutional的1X1输出)
    :param input_data:
    :param input_channel:
    :param filter_num1:
    :param filter_num2:
    :param trainable:
    :param name:
    :return:
    '''
    short_cut = input_data

    with tf.variable_scope(name):
        input_data = convolutional(input_data,filters_shape=(1,1,input_channel,filter_num1),
                                   trainable=trainable,name='conv1')# 1X1卷积操作

        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num1, filter_num2),
                                   trainable=trainable, name='conv2')# 3X3卷积操作

        residual_output = short_cut + input_data

    return residual_output


def route(name,previous_output,current_output):
    '''
    张量拼接操作
    :param name:
    :param previous_output:
    :param current_output:
    :return:
    '''

    with tf.variable_scope(name):
        output = tf.concat([previous_output,current_output],axis=-1)#对-1的理解：axis=-1表示对张量的最后一维进行拼接
    return output

def upsample(input_data,name,method='deconv'):
    '''
    上采样操作
    :param input_data:
    :param name:
    :param method:
    :return:
    '''
    assert method in ['resize','deconv']

    if method == 'resize':
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(images=input_data,size=(input_shape[1]*2,input_shape[2]*2))
            #images: A Tensor.4-D with shape [batch, height, width, channels]
            #size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new size for the images.
            #return: A Tensor. Has the same type as images.

    if method == 'deconv':
        num_filter = input_data.shape().as_list()[-1]#TODO:该行代码有疑问
        #反卷积操作or转置卷积
        output = tf.layers.conv2d_transpose(input_data,num_filter,kernel_size=2,
                                            padding='same',strides=(2,2),
                                            kernel_initializer=tf.random_normal_initializer())

    return output

