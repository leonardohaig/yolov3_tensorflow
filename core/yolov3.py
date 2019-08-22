#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:yolov3 的实现部分
#Date:2019.05.23
#Author:liheng
#Version:V1.0
#Reference:https://github.com/YunYang1994/tensorflow-yolov3/blob/master/core/yolov3.py
#============================#

__author__ = 'liheng'

import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config import cfg


class YOLOV3(object):
    '''
    Implement tensorflow yolove here
    '''

    def __init__(self,input_data,trainable):
        '''

        :param input_data:
        :param trainable:
        '''
        self.trainable = trainable
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)#类别列表
        self.num_class = len(self.classes)#类别数量
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.anchors = utils.get_anchors(cfg.YOLO.ANCHORS)#anchors
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh = cfg.YOLO.IOU_LOSS_THRESH
        self.upsample_method = cfg.YOLO.UPSAMPLE_METHOD#上采样方式

        try:
            self.conv_lbbox,self.conv_mbbox,self.conv_sbbox = self.__build_network(input_data)
        except:
            raise NotImplementedError("Can not build up yolov3 network!")

        with tf.variable_scope('pred_sbbox'):
            self.pred_sbbox = self.decode(self.conv_sbbox,self.anchors[0],self.strides[0])
        with tf.variable_scope('pred_mbbox'):
            self.pred_mbbox = self.decode(self.conv_mbbox,self.anchors[1],self.strides[1])
        with tf.variable_scope('pred_lbbox'):
            self.pred_lbbox = self.decode(self.conv_lbbox,self.anchors[2],self.strides[2])

    def __build_network(self, input_data):
        '''

        :param self:
        :param input_data:
        :return:
        '''

        # 输入层进入 Darknet-53 网络后，得到了三个分支
        route_1, route_2, input_data = backbone.darknet53(input_data, self.trainable)  # 输出的尺度为52,26,13
        # route_1.shape=(?,52,52,256)，一个特征点代表8*8的图像  检测小目标，最小检测8*8的图像
        # route_2.shape=(?,26,26,512)，一个特征点代表16*16的图像，检测中目标，最小检测16*16的图像
        # route_3.shape=(?,13,13,1024)， 一个特征点代表32*32像素的图像范围，可以用来检测大目标，最小检测32*32的图像。
        # ?具体大小与该批次输入的图片数量有关

        # ====predict one=======#
        # Convolutional Set 模块，1X1-->3X3-->1X1-->3X3-->1X1
        input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv52')
        input_data = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv53')
        input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv54')
        input_data = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv55')
        input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv56')

        # conv_lbbox 用于预测大尺寸物体，shape = [None, 13, 13, 255],255=3*(80+5)
        conv_lobj_branch = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, name='conv_lobj_branch')
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (self.num_class + 5)),
                                          trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)

        # 上采样
        # 这里的 upsample 使用的是最近邻插值方法，这样的好处在于上采样过程不需要学习，从而减少了网络参数
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv57')
        input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

        # 连接
        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)#input_data.shape(?,26,26,768),768=256(input_data)+512(route_2),第4维的拼接

        # ====predict two=======#
        # Convolutional Set 模块，1X1-->3X3-->1X1-->3X3-->1X1
        input_data = common.convolutional(input_data, (1, 1, 768, 256), self.trainable, 'conv58')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv59')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv60')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv61')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv62')

        # conv_mbbox 用于预测中等尺寸物体，shape = [None, 26, 26, 255]
        conv_mobj_branch = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, name='conv_mobj_branch')
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (self.num_class + 5)),
                                          trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, (1, 1,256, 128), self.trainable, 'conv63')
        input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)

        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)#input_data.shape(?,52,52,384),384=128(input_data)+256(route_1),第4维的拼接

        # ====predict three=======#
        # Convolutional Set 模块，1X1-->3X3-->1X1-->3X3-->1X1
        input_data = common.convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv64')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv65')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv66')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv67')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv68')

        # conv_sbbox 用于预测小尺寸物体，shape = [None, 52, 52, 255]
        conv_sobj_branch = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, name='conv_sobj_branch')
        conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (self.num_class + 5)),
                                          trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)

        return conv_lbbox, conv_mbbox, conv_sbbox

    def decode(self, conv_output, anchors, stride):
        '''
        return tensor of shape[batch_size,output_size,output_size,anchor_per_scale,5+num_classes] contains (x,y,w,h,score,probability)
        :param self:
        :param conv_output:
        :param anchors:
        :param stride:
        :return:
        '''

        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(conv_output,
                                 (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class))#可以这样理解，把输入的最后一维255拆开为： 3*(80+5)
        #意味着，输入从4维，增加了一维，如果输入为(1,52,52,255),经过该函数后，变为(1,52,52,3,85)

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]#应该表示 中心点？
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]#宽度、高度
        conv_raw_conf = conv_output[:, :, :, :, 4:5]#置信度
        conv_raw_prob = conv_output[:, :, :, :, 5:]#每一类别的概率

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def focal(self, target,actual,alpha=1,gamma=2):
        '''

        :param target:
        :param actual:
        :param alpha:
        :param gamma:
        :return:
        '''
        focal_loss = alpha * tf.pow(tf.abs(target-actual),gamma)
        return focal_loss

    def bbox_giou(self,boxes1,boxes2):
        '''

        :param boxes1:
        :param boxes2:
        :return:
        '''
        boxes1 = tf.concat([boxes1[..., :2]-boxes1[..., 2: ]*0.5,
                            boxes1[..., :2]+boxes1[..., 2: ]*0.5],axis=-1)
        boxes2 = tf.concat([boxes2[..., :2]-boxes2[..., 2: ]*0.5,
                            boxes2[..., :2]+boxes2[..., 2:]*0.5],axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2],boxes1[..., 2: ]),
                            tf.maximum(boxes1[..., :2],boxes1[..., 2: ])],axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2],boxes2[..., 2: ]),
                            tf.maximum(boxes2[..., :2],boxes2[..., 2: ])],axis=-1)

        boxes1_area = (boxes1[...,2]-boxes1[...,0]) * (boxes1[...,3]-boxes1[...,1])
        boxes2_area = (boxes2[...,2]-boxes2[...,0]) * (boxes2[...,3]-boxes2[...,1])

        left_up = tf.maximum(boxes1[..., :2],boxes2[..., :2])
        right_down = tf.minimum(boxes1[...,2: ],boxes2[...,2: ])

        inter_section = tf.maximum(right_down-left_up,0.0)
        inter_area = inter_section[...,0] * inter_section[...,1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area/union_area

        enclose_left_up = tf.minimum(boxes1[..., :2],boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[...,2: ],boxes2[...,2: ])
        enclose = tf.maximum(enclose_right_down-enclose_left_up,0.0)
        enclose_area = enclose[...,0] * enclose[...,1]
        giou = iou - 1.0*(enclose_area-union_area)/enclose_area

        return giou

    def bbox_iou(self,boxes1,boxes2):
        '''

        :param boxes1:
        :param boxes2:
        :return:
        '''

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2]-boxes1[...,2: ]*0.5,
                            boxes1[..., :2]+boxes1[...,2: ]*0.5],axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)


        left_up = tf.maximum(boxes1[..., :2],boxes2[..., :2])
        right_down = tf.minimum(boxes1[...,2: ],boxes2[...,2: ])

        inter_section = tf.maximum(right_down-left_up,0.0)
        inter_area = inter_section[...,0] * inter_section[...,1]
        union_area = boxes1_area + boxes2_area -inter_area
        iou = 1.0*inter_area/union_area

        return iou

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):
        '''

        :param conv:
        :param pred:
        :param label:
        :param bboxes:
        :param anchors:
        :param stride:
        :return:
        '''

        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        input_size = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.anchor_per_scale, 5 + self.num_class))
        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)

        conf_focal = self.focal(respond_bbox, pred_conf)

        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        return giou_loss, conf_loss, prob_loss

    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):
        '''

        :param label_sbbox:
        :param label_mbbox:
        :param label_lbbox:
        :param true_sbbox:
        :param true_mbbox:
        :param true_lbbox:
        :return:
        '''

        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox,
                                         anchors=self.anchors[0], stride=self.strides[0])

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         anchors=self.anchors[1], stride=self.strides[1])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         anchors=self.anchors[2], stride=self.strides[2])

        with tf.name_scope('giou_loss'):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        return giou_loss, conf_loss, prob_loss













