#!/usr/bin/env python3
#coding=utf-8

#============================#
#Update content:增加对3个pred_box的处理，通过置信度处理，nms过滤，使其返回为一个box，待验证
#Date:2019.10.27
#Author:liheng
#Version:V1.1

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

    def __init__(self, input_data, trainable,score_threshold=0.3,iou_threshold=0.45):
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
        self.per_cls_maxboxes = 200 # 一张图像上，每一类别的检测结果做大数量

        try:
            self.conv_lbbox,self.conv_mbbox,self.conv_sbbox = self.__build_network(input_data)
        except:
            raise NotImplementedError("Can not build up yolov3 network!")

        # 对输出加上name属性，这样在固化模型，生成.pb文件时，可以找到指定的节点

        with tf.variable_scope('pred_sbbox'):# 检测小目标，和anchor[0]对应
            self.pred_sbbox = self.decode(self.conv_sbbox,self.anchors[0],self.strides[0])# self.conv_sbbox.shaep:(batch_size,52,52,255)，strides[0]=8,   52*8=416(416/52=8)，一个特征点代表8*8的图像，检测小目标，对应的self.anchors[0]中的anchor宽高值必须是小目标的
        with tf.variable_scope('pred_mbbox'):
            self.pred_mbbox = self.decode(self.conv_mbbox,self.anchors[1],self.strides[1])
        with tf.variable_scope('pred_lbbox'):
            self.pred_lbbox = self.decode(self.conv_lbbox,self.anchors[2],self.strides[2])
        with tf.variable_scope('pred_res'): # 最终检测结果
            self.pred_res_boxes = self._get_pred_bboxes(input_data,score_threshold,iou_threshold)
            #预测结果框，shape[batchsize,num_class*per_cls_maxboxes,6]

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
        conv_raw_conf = conv_output[:, :, :, :, 4:5]#含有目标的置信度
        conv_raw_prob = conv_output[:, :, :, :, 5:]#每一类别的概率(如80个类别的各个概率)

        # x.shape / y.shape : (52,52) or (26,26) or (13,13)
        # x,y可以理解为 feature map 的坐标值，每个grid cell 左上角的横、纵坐标值,或者理解为画网格
        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        # 将x,y离散的坐标值合并在一起，方便计算，然后进行维度扩展，增加维度以和conv_output维度保持一致，方便数学运算求解
        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)#(52,52,2) or (26,26,2) or (13,13,2),2来源于坐标有两个：x，y
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])# 对张量进行扩展，扩展完毕后，shape:(?,52,52,3,2) or (?,26,26,3,2) or (?,13,13,3,2),? 与batch_size大小有关，3表示每一个特征图上预测3个anchor
        xy_grid = tf.cast(xy_grid, tf.float32)# 转换为float32类型

        # 矩形框坐标值从神经网络预测值(conv_raw_dxdy)转换为在feature_map上的位置，然后在转换到相对于416X416尺寸的图像上的位置
        # 该步骤为矩形框位置的预测，公式：
        # b_x = σ(t_x) + c_x
        # b_y = σ(t_y) + c_y
        # b_w = p_w * e^(t_w)
        # b_h = p_h * e^(t_h)
        # c_x,c_y是feature map中grid cell的左上角坐标;t_x,t_y是神经网络预测的边框中心坐标；t_w,t_h为预测的宽高
        # p_w,p_h是预设的anchor box映射到feature map中的宽和高，对于anchor=[10,13],
        # 映射到52*52的feature map上，其值为[10/8,13/8]=[1.25,1.625],其他类似。
        # 这就是data/anchors目录下，basline_anchors.txt文件的值比coco_anchors.txt小的原因，前者是后者经过映射后的值
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride #  *stride操作之后得到的坐标是相对于416X416尺寸的。
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        # 经过sigmoid函数后，其值约束在0~1之间
        pred_conf = tf.sigmoid(conv_raw_conf)# 存在目标的概率
        pred_prob = tf.sigmoid(conv_raw_prob)# 各个物体的概率

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

        :param conv:网络计算得到的结果
        :param pred:对conv进行decode得到的结果
        :param label:真实值？和pred.shape一致(batch_size,feature_map_size,feature_map_size,3,5+num_class)
        :param bboxes:真实值？shape(batch_size,?/150,4)150表示最多检测150个目标，4表示边框的坐标信息
        :param anchors:
        :param stride:
        :return:
        '''

        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        input_size = stride * output_size # 该值和图像大小一致，非原始图像大小，而是dataset.py中随机选择的图像大小
        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.anchor_per_scale, 5 + self.num_class))
        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]# 置信度，判断网格内有无物体
        label_prob = label[:, :, :, :, 5:]

        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        #GIou loss
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        # 找出与真实框 iou 值最大的预测框
        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        # 如果最大的 iou 小于阈值，那么认为该预测框不包含物体,则为背景框
        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)

        # 置信度损失
        conf_focal = self.focal(respond_bbox, pred_conf)
        # 计算置信度的损失（我们希望假如该网格中包含物体，那么网络输出的预测框置信度为 1，无物体时则为 0
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


    def get_pred_bboxes(self,input_data,score_threshold,iou_threshold):
        '''
        根据置信度和nms阈值，获取该批次数据的预测结果框
        :param input_data NHWC
        :param score_threshold:
        :param iou_threshold:
        :return:
        '''

        #取出batch中的1个image的检测结果进行处理
        def batch_map_fn(args):
            pred_sbbox,pred_mbbox,pred_lbbox = args

            pred_bbox = tf.concat([tf.reshape(pred_sbbox, (-1, 5 + self.num_class)),
                                   tf.reshape(pred_mbbox, (-1, 5 + self.num_class)),
                                   tf.reshape(pred_lbbox, (-1, 5 + self.num_class))],
                                  axis=0)  # pred_bbox.shape:(?,85)

            pred_xywh = pred_bbox[:, 0:4]  # 4列数据内容为：Center_x,Center_y,width,height(中心点坐标+宽高)
            pred_conf = pred_bbox[:, 4]  # 含有物体的概率
            pred_prob = pred_bbox[:, 5:]  # 各目标的概率

            # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
            pred_coor = tf.concat([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                   pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)

            # # (3) clip some boxes those are out of range
            input_image_h = tf.shape(input_data[0])[0]
            input_image_w = tf.shape(input_data[0])[1]

            pred_coor = tf.concat([tf.maximum(pred_coor[:, :2], [0, 0]),
                                   tf.minimum(pred_coor[:, 2:], [input_image_w - 1, input_image_h - 1])], axis=-1)
            invalid_mask = tf.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
            # pred_coor[invalid_mask] = 0
            # pred_coor1 = tf.where(invalid_mask,[[0,0,0,0]],pred_coor) # 对于mask位置处的坐标值，将值置0，其他位置保留原来的坐标值
            valid_mask = tf.logical_not(invalid_mask)

            # # (4) discard some invalid boxes
            valid_scale = [0, np.inf]
            bboxes_scale = tf.sqrt(
                tf.reduce_prod(pred_coor[:, 2:4] - pred_coor[:, 0:2], -1))  # √((xmax-xmin)*(ymax-ymin))
            scale_mask = tf.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))
            scale_mask = tf.logical_and(valid_mask, scale_mask)

            # # (5) discard some boxes with low scores
            classes = tf.argmax(pred_prob, axis=-1)  # 找出概率最大的class索引
            classes = tf.to_float(classes)
            max_value = tf.reduce_max(pred_prob, reduction_indices=[1])  # 找出行上最大值，即找出概率最大的class
            scores = pred_conf * max_value
            score_mask = scores > score_threshold
            mask = tf.logical_and(scale_mask, score_mask)
            coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

            # 合并结果
            bboxes = tf.concat([coors, scores[:, tf.newaxis], classes[:, tf.newaxis]],
                               axis=-1)  # [xmin,ymin,xmax,ymax,prob,classid]

            # ===============nms过滤=======================#
            def nms_map_fn(args):
                '''

                :param args:
                :return:
                '''

                cls = args
                cls = tf.cast(cls, dtype=tf.int32)

                _bboxes = tf.cast(bboxes[:, 5], dtype=tf.int32)  # 类别ID
                cls_mask = tf.equal(_bboxes, cls)
                cls_bboxes = bboxes[cls_mask]  # ID为cls的目标框

                # 拆分得到boxes，scores，以便调用tf.image.non_max_suppression
                # nms之后再来合并
                # https://cloud.tencent.com/developer/article/1486383
                boxes = cls_bboxes[:, 0:4]
                scores = cls_bboxes[:, 4]
                _maxbox = tf.shape(scores)[0]  # nms操作最多输出多少个目标

                selected_indices = tf.image.non_max_suppression(boxes=boxes, scores=scores,
                                                                iou_threshold=iou_threshold,
                                                                max_output_size=_maxbox)
                selected_boxes = tf.gather(boxes, selected_indices)
                seclected_scores = tf.gather(scores, selected_indices)
                # classes = tf.Variable(tf.fill([objnum, 1], cls))
                classes = tf.ones_like(seclected_scores, dtype=tf.int32) * cls
                # classes = tf.where(seclected_scores>-100, seclected_scores, seclected_scores)
                classes = tf.to_float(classes)

                selected_bboxes = tf.concat([selected_boxes,
                                             seclected_scores[:, tf.newaxis],
                                             classes[:, tf.newaxis]],
                                            axis=-1)  # [xmin,ymin,xmax,ymax,prob,classid]

                # selected_bboxes = selected_bboxes[tf.argsort(tf.cast(selected_bboxes[:, 4]*1000,dtype=tf.int32),direction='DESCENDING')] #根据概率降序排序

                objnum = tf.shape(selected_boxes)[0]  # nms得到的目标数量
                selected_bboxes = selected_bboxes[:self.per_cls_maxboxes]

                def add_boxes():
                    temp_bboxes = tf.fill([self.per_cls_maxboxes - objnum, 6], -1)  # 创建一个常量
                    temp_bboxes = tf.to_float(temp_bboxes)
                    _selected_bboxes = tf.concat([selected_bboxes, temp_bboxes], axis=0)
                    return _selected_bboxes

                def ori_boxes():
                    return selected_bboxes

                selected_bboxes = tf.cond(objnum < self.per_cls_maxboxes, true_fn=add_boxes, false_fn=ori_boxes)

                return selected_bboxes

            classes_in_img, idx = tf.unique(tf.cast(bboxes[:, 5],tf.int32))
            best_bboxes = tf.map_fn(nms_map_fn, classes_in_img, infer_shape=False, dtype=tf.float32)

            #填充行数与类别数一致
            clsnum = tf.shape(best_bboxes)[0]
            best_bboxes = best_bboxes[:self.num_class]
            def add_classes():
                temp_classes = tf.fill([self.num_class - clsnum, self.num_class, 6], -1)  # 创建一个常量
                temp_classes = tf.to_float(temp_classes)
                _best_bboxes = tf.concat([best_bboxes, temp_classes], axis=0)
                return _best_bboxes

            def ori_classes():
                return best_bboxes

            best_bboxes = tf.cond(clsnum < self.num_class, true_fn=add_classes, false_fn=ori_classes)

            # 给变量一名称
            # best_bboxes = tf.add_n([best_bboxes], name='pred_bboxes')
            return best_bboxes


        best_bboxes = tf.map_fn(batch_map_fn,(self.pred_sbbox,self.pred_mbbox,self.pred_lbbox),dtype=tf.float32,infer_shape=False)

        # 给变量一名称
        best_bboxes = tf.add_n([best_bboxes], name='pred_bboxes')

        return best_bboxes






        pred_bbox = tf.concat([tf.reshape(self.pred_sbbox, (-1, 5 + self.num_class)),
                               tf.reshape(self.pred_mbbox, (-1, 5 + self.num_class)),
                               tf.reshape(self.pred_lbbox, (-1, 5 + self.num_class))],
                              axis=0) # pred_bbox.shape:(?,85)

        pred_xywh = pred_bbox[:, 0:4]  # 4列数据内容为：Center_x,Center_y,width,height(中心点坐标+宽高)
        pred_conf = pred_bbox[:, 4]  # 含有物体的概率
        pred_prob = pred_bbox[:, 5:]  # 各目标的概率

        # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = tf.concat([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                               pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)

        # # (3) clip some boxes those are out of range
        input_image_h = tf.shape(input_data[0])[0]
        input_image_w = tf.shape(input_data[0])[1]

        pred_coor = tf.concat([tf.maximum(pred_coor[:, :2], [0, 0]),
                               tf.minimum(pred_coor[:, 2:], [input_image_w - 1, input_image_h - 1])],axis=-1)
        invalid_mask = tf.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        # pred_coor[invalid_mask] = 0
        # pred_coor1 = tf.where(invalid_mask,[[0,0,0,0]],pred_coor) # 对于mask位置处的坐标值，将值置0，其他位置保留原来的坐标值
        valid_mask = tf.logical_not(invalid_mask)

        # # (4) discard some invalid boxes
        valid_scale = [0, np.inf]
        bboxes_scale = tf.sqrt(tf.reduce_prod(pred_coor[:, 2:4] - pred_coor[:, 0:2], -1)) # √((xmax-xmin)*(ymax-ymin))
        scale_mask = tf.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))
        scale_mask = tf.logical_and(valid_mask,scale_mask)


        # # (5) discard some boxes with low scores
        classes = tf.argmax(pred_prob, axis=-1) # 找出概率最大的class索引
        classes = tf.to_float(classes)
        max_value = tf.reduce_max(pred_prob, reduction_indices=[1])#找出行上最大值，即找出概率最大的class
        scores = pred_conf * max_value
        score_mask = scores > score_threshold
        mask = tf.logical_and(scale_mask, score_mask)
        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

        # 合并结果
        bboxes = tf.concat([coors, scores[:, tf.newaxis], classes[:, tf.newaxis]], axis=-1) # [xmin,ymin,xmax,ymax,prob,classid]

        #===============nms过滤=======================#
        def nms_map_fn(args):
            '''

            :param args:
            :return:
            '''

            cls = args
            cls = tf.cast(cls,dtype=tf.int32)

            _bboxes = tf.cast(bboxes[:, 5],dtype=tf.int32) # 类别ID
            cls_mask = tf.equal(_bboxes,cls)
            cls_bboxes = bboxes[cls_mask] # ID为cls的目标框

            # 拆分得到boxes，scores，以便调用tf.image.non_max_suppression
            # nms之后再来合并
            # https://cloud.tencent.com/developer/article/1486383
            boxes = cls_bboxes[:, 0:4]
            scores = cls_bboxes[:, 4]
            _maxbox = tf.shape(scores)[0]  # nms操作最多输出多少个目标

            selected_indices = tf.image.non_max_suppression(boxes=boxes, scores=scores,
                                                            iou_threshold=iou_threshold,
                                                            max_output_size=_maxbox)
            selected_boxes = tf.gather(boxes, selected_indices)
            seclected_scores = tf.gather(scores, selected_indices)
            #classes = tf.Variable(tf.fill([objnum, 1], cls))
            classes = tf.ones_like(seclected_scores,dtype=tf.int32) * cls
            #classes = tf.where(seclected_scores>-100, seclected_scores, seclected_scores)
            classes = tf.to_float(classes)

            selected_bboxes = tf.concat([selected_boxes,
                                         seclected_scores[:, tf.newaxis],
                                         classes[:, tf.newaxis]],
                                        axis=-1)  # [xmin,ymin,xmax,ymax,prob,classid]

            # selected_bboxes = selected_bboxes[tf.argsort(tf.cast(selected_bboxes[:, 4]*1000,dtype=tf.int32),direction='DESCENDING')] #根据概率降序排序

            objnum = tf.shape(selected_boxes)[0]  # nms得到的目标数量
            selected_bboxes = selected_bboxes[:self.per_cls_maxboxes]

            def add_boxes():
                temp_bboxes = tf.fill([self.per_cls_maxboxes - objnum, 6], -1)  # 创建一个常量
                temp_bboxes = tf.to_float(temp_bboxes)
                _selected_bboxes = tf.concat([selected_bboxes, temp_bboxes], axis=0)
                return _selected_bboxes
            def ori_boxes():
                return selected_bboxes


            selected_bboxes = tf.cond(objnum<self.per_cls_maxboxes,true_fn=add_boxes,false_fn=ori_boxes)

            return selected_bboxes

        classes_in_img, idx = tf.unique(bboxes[:, 5])
        best_bboxes = tf.map_fn(nms_map_fn,classes_in_img,infer_shape=False) #infer_shape=False 禁用对一致输出形状的测试

        # 给变量一名称
        best_bboxes = tf.add_n([best_bboxes], name='pred_bboxes') # 暂时忽略
        return best_bboxes



    def _get_pred_bboxes(self,input_data,score_threshold,iou_threshold):
        '''
        根据置信度和nms阈值，获取该批次数据的预测结果框
        :param input_data NHWC
        :param score_threshold:
        :param iou_threshold:
        :return:
        '''

        #取出batch中的1个image的检测结果进行处理
        def batch_map_fn(args):
            pred_sbbox,pred_mbbox,pred_lbbox = args

            pred_bbox = tf.concat([tf.reshape(pred_sbbox, (-1, 5 + self.num_class)),
                                   tf.reshape(pred_mbbox, (-1, 5 + self.num_class)),
                                   tf.reshape(pred_lbbox, (-1, 5 + self.num_class))],
                                  axis=0)  # pred_bbox.shape:(?,85)

            pred_xywh = pred_bbox[:, 0:4]  # 4列数据内容为：Center_x,Center_y,width,height(中心点坐标+宽高)
            pred_conf = pred_bbox[:, 4]  # 含有物体的概率
            pred_prob = pred_bbox[:, 5:]  # 各目标的概率

            # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
            pred_coor = tf.concat([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                   pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)

            # # (3) clip some boxes those are out of range
            input_image_h = tf.shape(input_data[0])[0]
            input_image_w = tf.shape(input_data[0])[1]

            pred_coor = tf.concat([tf.maximum(pred_coor[:, :2], [0, 0]),
                                   tf.minimum(pred_coor[:, 2:], [input_image_w - 1, input_image_h - 1])], axis=-1)
            invalid_mask = tf.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
            # pred_coor[invalid_mask] = 0
            # pred_coor1 = tf.where(invalid_mask,[[0,0,0,0]],pred_coor) # 对于mask位置处的坐标值，将值置0，其他位置保留原来的坐标值
            valid_mask = tf.logical_not(invalid_mask)

            # # (4) discard some invalid boxes
            valid_scale = [0, np.inf]
            bboxes_scale = tf.sqrt(
                tf.reduce_prod(pred_coor[:, 2:4] - pred_coor[:, 0:2], -1))  # √((xmax-xmin)*(ymax-ymin))
            scale_mask = tf.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))
            scale_mask = tf.logical_and(valid_mask, scale_mask)

            # # (5) discard some boxes with low scores
            classes = tf.argmax(pred_prob, axis=-1)  # 找出概率最大的class索引
            classes = tf.to_float(classes)
            max_value = tf.reduce_max(pred_prob, reduction_indices=[1])  # 找出行上最大值，即找出概率最大的class
            scores = pred_conf * max_value
            score_mask = scores > score_threshold
            mask = tf.logical_and(scale_mask, score_mask)
            coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

            # 合并结果
            bboxes = tf.concat([coors, scores[:, tf.newaxis], classes[:, tf.newaxis]],
                               axis=-1)  # [xmin,ymin,xmax,ymax,prob,classid]

            # ===============nms过滤=======================#
            def nms_map_fn(args):
                '''

                :param args:
                :return:
                '''

                cls = args
                cls = tf.cast(cls, dtype=tf.int32)

                _bboxes = tf.cast(bboxes[:, 5], dtype=tf.int32)  # 类别ID
                cls_mask = tf.equal(_bboxes, cls)
                cls_bboxes = bboxes[cls_mask]  # ID为cls的目标框

                # 拆分得到boxes，scores，以便调用tf.image.non_max_suppression
                # nms之后再来合并
                # https://cloud.tencent.com/developer/article/1486383
                boxes = cls_bboxes[:, 0:4]
                scores = cls_bboxes[:, 4]
                _maxbox = tf.shape(scores)[0]  # nms操作最多输出多少个目标

                selected_indices = tf.image.non_max_suppression(boxes=boxes, scores=scores,
                                                                iou_threshold=iou_threshold,
                                                                max_output_size=_maxbox)
                selected_boxes = tf.gather(boxes, selected_indices)
                seclected_scores = tf.gather(scores, selected_indices)
                classes = tf.ones_like(seclected_scores, dtype=tf.int32) * cls
                classes = tf.to_float(classes)

                selected_bboxes = tf.concat([selected_boxes,
                                             seclected_scores[:, tf.newaxis],
                                             classes[:, tf.newaxis]],
                                            axis=-1)  # [xmin,ymin,xmax,ymax,prob,classid]

                # selected_bboxes = selected_bboxes[tf.argsort(tf.cast(selected_bboxes[:, 4]*1000,dtype=tf.int32),direction='DESCENDING')] #根据概率降序排序

                objnum = tf.shape(selected_boxes)[0]  # nms得到的目标数量
                selected_bboxes = selected_bboxes[:self.per_cls_maxboxes]

                def add_boxes():
                    temp_bboxes = tf.fill([self.per_cls_maxboxes - objnum, 6], -1)  # 创建一个常量
                    temp_bboxes = tf.to_float(temp_bboxes)
                    _selected_bboxes = tf.concat([selected_bboxes, temp_bboxes], axis=0)
                    return _selected_bboxes

                def ori_boxes():
                    return selected_bboxes

                selected_bboxes = tf.cond(objnum < self.per_cls_maxboxes, true_fn=add_boxes, false_fn=ori_boxes)

                return selected_bboxes

            classes_in_img, idx = tf.unique(tf.cast(bboxes[:, 5],tf.int32))
            best_bboxes = tf.cond(tf.equal(tf.size(classes_in_img),0), # 防止类别为空
                                  false_fn=lambda:tf.map_fn(nms_map_fn, classes_in_img,
                                                           infer_shape=False, dtype=tf.float32),
                                  true_fn=lambda:tf.to_float(tf.fill([self.num_class,self.per_cls_maxboxes,6],-1))
                                  )

            #填充行数与类别数一致
            clsnum = tf.shape(best_bboxes)[0]
            best_bboxes = best_bboxes[:self.num_class]
            def add_classes():
                temp_classes = tf.fill([self.num_class - clsnum, self.per_cls_maxboxes, 6], -1)  # 创建一个常量
                temp_classes = tf.to_float(temp_classes)
                _best_bboxes = tf.concat([best_bboxes, temp_classes], axis=0)
                return _best_bboxes

            def ori_classes():
                return best_bboxes

            best_bboxes = tf.cond(clsnum < self.num_class, true_fn=add_classes, false_fn=ori_classes)

            # 给变量一名称
            # best_bboxes = tf.add_n([best_bboxes], name='pred_bboxes')
            return best_bboxes


        best_bboxes = tf.map_fn(batch_map_fn,(self.pred_sbbox,self.pred_mbbox,self.pred_lbbox),dtype=tf.float32,infer_shape=False)
        N = tf.shape(best_bboxes)[0]
        cls = tf.shape(best_bboxes)[1]
        maxbox = tf.shape(best_bboxes)[2]

        best_bboxes = tf.reshape(best_bboxes,[N,cls*maxbox,6],name='pred_bboxes')

        return best_bboxes

    def get_imgage_predbboxes(self):
        return self.pred_res_boxes


    def get_pred_image(self,input_data):
        '''
        在py_func中不能定义可训练的参数参与网络训练(反传)
        :param input_data:
        :return:
        '''

        self.pred_image = tf.py_function(utils.draw_batch_bbox,[input_data,self.pred_res_boxes],tf.float32)
        self.pred_image.set_shape([None, None,
                              None, 3])

        return self.pred_image
