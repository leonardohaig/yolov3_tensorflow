#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:config.py
#Date:2019.05.20
#Author:liheng
#Version:V1.0
#Reference:https://github.com/YunYang1994/tensorflow-yolov3/blob/master/core/config.py
#============================#

__author__ = 'liheng'

from easydict import EasyDict as edict

__C                             = edict()

cfg                             = __C

#YOLO options
__C.YOLO                        = edict()
__C.YOLO.CLASSES                = './data/classes/bdd100k.names' #类别名称存放路径
#__C.YOLO.CLASSES                = './data/classes/raccoon.names' #类别名称存放路径
__C.YOLO.ANCHORS                = './data/anchors/bdd100k_basline_anchors.txt' # anchors比例存放路径
__C.YOLO.MOVING_AVE_DECAY       = 0.995 # 滑动平均时的decay值
__C.YOLO.STRIDES                = [8,16,32]
__C.YOLO.ANCHOR_PER_SCALE       = 3 # 每个尺度上anchor窗口的数量
__C.YOLO.IOU_LOSS_THRESH        = 0.5
__C.YOLO.UPSAMPLE_METHOD        = 'resize' #上采样方式，'resize' or `deconv`
__C.YOLO.ORIGINAL_WEIGHT        = './checkpoint/yolov3_coco.ckpt'
__C.YOLO.DEMO_WEIGHT            = './checkpoint/yolov3_coco_demo.ckpt'


#Train options
__C.TRAIN                       = edict()
#__C.TRAIN.ANNOT_PATH            = './data/dataset/voc_train.txt'
__C.TRAIN.ANNOT_PATH            = './tools/bdd100k_train.txt' # 训练集各图像及对应的标签、目标位置
__C.TRAIN.BATCH_SIZE            = 10
__C.TRAIN.INPUT_SIZE            = [416,448,480,512,544,576,608] #训练时，输入图片尺寸。多尺度训练
__C.TRAIN.DATA_AUG              = False # 是否进行数据增强，data augmentation,以扩大数据集（对原始图像进行裁剪、平移等变换）
__C.TRAIN.LEARN_RATE_INIT       = 1e-4 #初始学习率
__C.TRAIN.LEARN_RATE_END        = 1e-6 #终学习率
__C.TRAIN.WARMUP_EPOCHS         = 5
__C.TRAIN.FISRT_STAGE_EPOCHS    = 0 #初始迭代次数,第一阶段训练：仅仅训练三个分支的最后卷积层
__C.TRAIN.SECOND_STAGE_EPOCHS   = 50000 #终迭代次数,第二阶段训练,训练所有层
__C.TRAIN.MODEL_SAVE_DIR        = '/home/liheng/liheng/bdd100k_mobilenet_checkpoint' #ckpt文件保存路径,同时，开始训练时自动从该路径加载最近一次训练结果
__C.TRAIN.LOG_SAVE_DIR          = './data/log/' # 训练日志保存路径,程序会自动删除该路径并重建，务必确保该文件夹为空

#Test options
__C.TEST                        = edict()
#__C.TEST.ANNOT_PATH             = './data/dataset/voc_test.txt'
__C.TEST.ANNOT_PATH             = './tools/bdd100k_val.txt' # 验证集各图像及对应的标签、目标位置
__C.TEST.BATCH_SIZE             = 8
__C.TEST.INPUT_SIZE             = 416
__C.TEST.DATA_AUG               = False # 是否进行数据增强（对原始图像进行裁剪、平移等变换）
__C.TEST.WRITE_IMAGE            = True
__C.TEST.WRITE_IMAGE_PATH       = './data/detection/'
__C.TEST.WRITE_IMAGE_SHOW_LABEL = False
__C.TEST.WEIGHT_FILE            = './checkpoint/yolov3_test_loss=9.2099.ckpt-5'
__C.TEST.SHOW_LABEL             = True
__C.TEST.SCORE_THRESHOLD        = 0.2
__C.TEST.IOU_THRESHOLD          = 0.45

