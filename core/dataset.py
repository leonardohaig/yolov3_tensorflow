#!/usr/bin/env python3
#coding=utf-8

#============================#
#Update content:采用线程和队列进行数据读取和存储
#Date:2019.12.7
#Author:liheng
#Version:V1.2

#Update content:返回批数据时，增加ground truth box image的显示
#Date:2019.10.26
#Author:liheng
#Version:V1.1
#Program:dataset.py
#Date:2019.06.05
#Author:liheng
#Version:V1.0
#Reference:https://github.com/YunYang1994/tensorflow-yolov3/blob/master/core/dataset.py
#============================#

__author__ = 'liheng'

import os

if __name__ == '__main__':#测试dataset类用，此时需要切换工作目录为上一级
    os.chdir('../')

import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from queue import Queue
import queue
import threading

from multiprocessing import Pool



class Dataset(object):
    """implement Dataset here"""
    def __init__(self, dataset_type):
        '''

        :param dataset_type:
        '''
        self.images_dir  = cfg.TRAIN.IMAGES_DIR if dataset_type == 'train' else cfg.TEST.IMAGES_DIR
        self.annot_path  = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES) # dict类型，ID--name
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150#每个尺度最多检测目标数量

        self.annotations = self.load_annotations()#训练/验证文件的每一行
        self.num_samples = len(self.annotations)#训练集/验证集的数量
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))#训练用/验证用批次数量
        self.batch_count = 0

        self.__data_queue = Queue(maxsize=10)#通过队列储存数据
        self.__bQuitDataThread = False #保证可以在循环中不断读取数据，防止Queue() Full 不能结束线程
        self.__data_thread = threading.Thread(target=self.__thread_readData)#线程不断读取数据
        self.__data_thread.start()#开启线程


    def load_annotations(self):
        '''

        :param dataset_type:
        :return:类型：list，值：文件路径中的每一行
        '''
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
            annotations = [os.path.join(self.images_dir,line) for line in annotations]
        np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __thread_readData(self):
        '''
        该线程不断读取数据，储存于队列中，生产者线程
        :return:
        '''
        idx=0
        while (idx < self.num_batchs and not self.__bQuitDataThread):
            with tf.device('/cpu:0'):  # 指定设备为cpu，意味着以下操作在cpu上完成
                train_input_size = random.choice(self.train_input_sizes)  # 网络输入图像尺寸，随机选择，非固定的416X416，
                train_output_sizes = train_input_size // self.strides  # 网络输出图像尺寸，指定输出尺寸，可以保证权重矩阵维度一致

                batch_image = np.zeros((self.batch_size, train_input_size, train_input_size, 3))
                batch_gt_image = np.zeros((self.batch_size, train_input_size, train_input_size, 3),
                                          dtype=np.float32)  # 绘制有ground truth的图像

                batch_label_sbbox = np.zeros((self.batch_size, train_output_sizes[0], train_output_sizes[0],
                                              self.anchor_per_scale, 5 + self.num_classes))
                batch_label_mbbox = np.zeros((self.batch_size, train_output_sizes[1], train_output_sizes[1],
                                              self.anchor_per_scale, 5 + self.num_classes))
                batch_label_lbbox = np.zeros((self.batch_size, train_output_sizes[2], train_output_sizes[2],
                                              self.anchor_per_scale, 5 + self.num_classes))

                batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
                batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
                batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))

                #读取数据，构造batch
                num = 0
                # p = Pool(self.batch_size)
                # while num < self.batch_size:
                for num in range(self.batch_size):
                    # p.apply_async(self.__construct_batch,
                    #               args=(idx, num, train_input_size, train_output_sizes,
                    #                     batch_image,
                    #                     batch_label_sbbox, batch_label_mbbox, batch_label_lbbox,
                    #                     batch_sbboxes, batch_mbboxes, batch_lbboxes,
                    #                     batch_gt_image))


                    index = idx * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation,train_input_size)  # 读取图像，box(位置+类别)，图像已经经过预处理、数据增强等变换
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(
                        bboxes,train_output_sizes)

                    # 在image标出true box及其类别
                    _temp = np.ones(bboxes.shape[0])  # 作为置信度
                    _bboxes = np.insert(bboxes, 4, values=_temp, axis=1)  # 增加一列
                    gt_image = utils.draw_bbox(image * 255, _bboxes)  # 输入image类型需要为uint8类型(0-255之间),
                    batch_gt_image[num, :, :, :] = gt_image / 255.0  # 归一化0-1区间，类型为float32

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    # num += 1


                # p.close()
                # p.join()

                #将数据存放于队列中
                try:
                    self.__data_queue.put([batch_image,
                                           batch_label_sbbox, batch_label_mbbox, batch_label_lbbox,
                                           batch_sbboxes, batch_mbboxes, batch_lbboxes,
                                           batch_gt_image], timeout=10) #10s内仍为Full状态，
                                        # 考虑程序可能是已经运行结束了，需要退出该线程了，到while处进行判断是否要退出
                    idx += 1
                except queue.Full:
                    pass # 针对Full的异常不用进行处理，只用继续循环即可

        # print("Read data complete !")
    def stopThread(self):
        self.__bQuitDataThread = True
        self.__data_thread.join()

    def __next__(self):# 返回下一批次的数据
        with tf.device('/cpu:0'):# 指定设备为cpu，意味着以下操作在cpu上完成
            if self.batch_count < self.num_batchs:
                self.batch_count += 1
                return self.__data_queue.get()
            else:
                self.__data_thread.join()
                self.batch_count = 0
                np.random.shuffle(self.annotations)

                #重新开启线程
                self.__data_thread = threading.Thread(target=self.__thread_readData)  # 线程不断读取数据
                self.__data_thread.start()  # 开启线程
                raise StopIteration
    def __construct_batch(self,cur_batch, idx, train_input_size, train_output_sizes,
                          batch_image,
                          batch_label_sbbox, batch_label_mbbox, batch_label_lbbox,
                          batch_sbboxes, batch_mbboxes, batch_lbboxes,
                          batch_gt_image):
        """

        :param cur_batch:
        :param idx:
        :param train_input_size:
        :param train_output_sizes:
        :param batch_image:
        :param batch_label_sbbox:
        :param batch_label_mbbox:
        :param batch_label_lbbox:
        :param batch_sbboxes:
        :param batch_mbboxes:
        :param batch_lbboxes:
        :param batch_gt_image:
        :return:
        """
        index = cur_batch * self.batch_size + idx
        if index >= self.num_samples: index -= self.num_samples
        annotation = self.annotations[index]
        image, bboxes = self.parse_annotation(annotation, train_input_size)  # 读取图像，box(位置+类别)，图像已经经过预处理、数据增强等变换
        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(
            bboxes, train_output_sizes)

        # 在image标出true box及其类别
        _temp = np.ones(bboxes.shape[0])  # 作为置信度
        _bboxes = np.insert(bboxes, 4, values=_temp, axis=1)  # 增加一列
        gt_image = utils.draw_bbox(image * 255, _bboxes)  # 输入image类型需要为uint8类型(0-255之间),
        batch_gt_image[idx, :, :, :] = gt_image / 255.0  # 归一化0-1区间，类型为float32

        batch_image[idx, :, :, :] = image
        batch_label_sbbox[idx, :, :, :, :] = label_sbbox
        batch_label_mbbox[idx, :, :, :, :] = label_mbbox
        batch_label_lbbox[idx, :, :, :, :] = label_lbbox
        batch_sbboxes[idx, :, :] = sbboxes
        batch_mbboxes[idx, :, :] = mbboxes
        batch_lbboxes[idx, :, :] = lbboxes


    # def __next__(self):# 返回下一批次的数据
    #
    #     with tf.device('/cpu:0'):# 指定设备为cpu，意味着以下操作在cpu上完成
    #         train_input_size = random.choice(self.train_input_sizes) # 网络输入图像尺寸，随机选择，非固定的416X416，
    #         train_output_sizes = train_input_size // self.strides # 网络输出图像尺寸，指定输出尺寸，可以保证权重矩阵维度一致
    #
    #         batch_image = np.zeros((self.batch_size, train_input_size, train_input_size, 3))
    #         batch_gt_image = np.zeros((self.batch_size, train_input_size, train_input_size, 3),dtype=np.float32)#绘制有ground truth的图像
    #
    #         batch_label_sbbox = np.zeros((self.batch_size, train_output_sizes[0], train_output_sizes[0],
    #                                       self.anchor_per_scale, 5 + self.num_classes))
    #         batch_label_mbbox = np.zeros((self.batch_size, train_output_sizes[1], train_output_sizes[1],
    #                                       self.anchor_per_scale, 5 + self.num_classes))
    #         batch_label_lbbox = np.zeros((self.batch_size, train_output_sizes[2], train_output_sizes[2],
    #                                       self.anchor_per_scale, 5 + self.num_classes))
    #
    #         batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
    #         batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
    #         batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
    #
    #         num = 0
    #         if self.batch_count < self.num_batchs:
    #             while num < self.batch_size:
    #                 index = self.batch_count * self.batch_size + num
    #                 if index >= self.num_samples: index -= self.num_samples
    #                 annotation = self.annotations[index]
    #                 image, bboxes = self.parse_annotation(annotation,train_input_size) # 读取图像，box(位置+类别)，图像已经经过预处理、数据增强等变换
    #                 label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes,train_output_sizes)
    #
    #                 # 在image标出true box及其类别
    #                 _temp = np.ones(bboxes.shape[0])#作为置信度
    #                 _bboxes = np.insert(bboxes,4,values=_temp,axis=1)#增加一列
    #                 gt_image = utils.draw_bbox(image*255,_bboxes) #输入image类型需要为uint8类型(0-255之间),
    #                 batch_gt_image[num, :, :, :] = gt_image/255.0 # 归一化0-1区间，类型为float32
    #
    #                 batch_image[num, :, :, :] = image
    #                 batch_label_sbbox[num, :, :, :, :] = label_sbbox
    #                 batch_label_mbbox[num, :, :, :, :] = label_mbbox
    #                 batch_label_lbbox[num, :, :, :, :] = label_lbbox
    #                 batch_sbboxes[num, :, :] = sbboxes
    #                 batch_mbboxes[num, :, :] = mbboxes
    #                 batch_lbboxes[num, :, :] = lbboxes
    #                 num += 1
    #             self.batch_count += 1
    #             return batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
    #                    batch_sbboxes, batch_mbboxes, batch_lbboxes,batch_gt_image
    #         else:
    #             self.batch_count = 0
    #             np.random.shuffle(self.annotations)
    #             raise StopIteration

    def random_horizontal_flip(self, image, bboxes):
        '''

        :param image:
        :param bboxes:
        :return:
        '''
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

        return image, bboxes

    def random_crop(self, image, bboxes):
        '''

        :param image:
        :param bboxes:
        :return:
        '''
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):
        '''

        :param image:
        :param bboxes:
        :return:
        '''
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def parse_annotation(self, annotation,train_input_size):
        '''

        :param annotation:类型：list
        :return:
        '''
        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
        image = np.array(cv2.imread(image_path))#读取图像
        bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])

        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        image, bboxes = utils.image_preprocess(np.copy(image), [train_input_size, train_input_size], np.copy(bboxes))
        return image, bboxes

    def bbox_iou(self, boxes1, boxes2):
        '''

        :param boxes1:
        :param boxes2:
        :return:
        '''
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    def preprocess_true_boxes(self, bboxes,train_output_sizes):
        '''

        :param bboxes:
        :return:
        '''
        label = [np.zeros((train_output_sizes[i], train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        # label 为list类型，共有3个元素，
        # 每个元素的shape为[train_output_sizes[i],train_output_sizes[i],
        # self.anchor_per_scale,5 + self.num_classes]

        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)] # 这里 4 来源于坐标有 x,y,w,h 四个值（可以理解为4列），
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4] # 位置
            bbox_class_ind = bbox[4] # 类别

            onehot = np.zeros(self.num_classes, dtype=np.float) # 理解为1列数组，共80行，某一行的值为1，意味着为该类别。例：oneshot[2]=1.0,意味着该矩形框物体属于类别2
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution# 应是将置信度1.0进行平滑处理，保证在0-1区间内

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)# 将边框的顶点坐标转换为中心点+宽高的形式
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis] # 将边框坐标映射到3个feature map上

            iou = []
            exist_positive = False
            for i in range(3): # 3种网络尺寸
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))#每个尺度下每个点预测3个anchor box，4表示中心位置和宽高
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5# 中心点坐标，来源于ground truth 的在3个feature map上的映射
                anchors_xywh[:, 2:4] = self.anchors[i] #宽高来源于预设的anchor在feature map上的映射

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3 # 判断其和真实的box的iou是否>0.3

                if np.any(iou_mask):# 针对 iou > 0.3 的 anchor 框进行处理.一个物体可能同时在多个feature map上匹配到，以及被一个feature map上的多个anchor匹配到
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)# 中心点坐标 #根据真实框的坐标信息来计算所属网格左上角的位置

                    label[i][yind, xind, iou_mask, :] = 0 # 将无关的feature map上置0，iou_mask中值为False的label对应列为0，
                    # 填充真实框的中心位置和宽高
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh # TODO：此处应是bbox_xywh_scaled[i]吧? 经过查看后面计算损失函数部分，应该是直接利用坐标值，而非缩放、归一化后的坐标值
                    label[i][yind, xind, iou_mask, 4:5] = 1.0 # 设定置信度为 1.0，表明该网格包含物体
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot#平滑处理,具体为某个物体的概率

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh # TODO：这里如何理解？
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs


if __name__ == '__main__':
    '''
    测试数据的读取，同时将true box显示出来
    '''
    trainset = Dataset('train')

    nWaitTime = 0
    for i in range(5):
        for train_data in trainset:
            # batch_bboxes = np.random.randint(10,50,size=(cfg.TRAIN.BATCH_SIZE,20,6)).astype(np.float)
            # batch_bboxes[:,10:20] = -1
            # utils.draw_batch_bbox(train_data[0],batch_bboxes)

            cv2.imshow("gt_image", train_data[7][0])  # 仅展示1个batch中的第一幅图像

            key = cv2.waitKey(nWaitTime)
            if 27 == key:  # ESC
                break
            elif 32 == key:  # space
                nWaitTime = not nWaitTime

    trainset.stopThread()
