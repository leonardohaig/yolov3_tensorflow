#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:utils.py
#Date:2019.05.22
#Author:liheng
#Version:V1.0
#Reference:https://github.com/YunYang1994/tensorflow-yolov3/blob/master/core/utils.py
#============================#

__author__ = 'liheng'

import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
from core.config import cfg


def read_class_names(class_file_name):
    '''
    load classes's name from the given file
    :param class_file_name:
    :return:返回值类型：dict,ID--类别名称
    '''
    names = {}
    with open(class_file_name,'r') as data:
        for ID,name in enumerate(data):
            names[ID] = name.strip('\n')#去除每一行结尾的换行符

    return names

def get_anchors(anchors_path):
    '''

    :param anchors_path:
    :return:
    '''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','),dtype=np.float32)
    return anchors.reshape(3,3,2)



def image_preprocess(image,target_size,gt_boxes=None):
    '''
    将图像及groundtruth boxes进行缩放，以及图像进行归一化
    :param image:
    :param target_size ;data type:list
    :param gt_boxes:
    :return:
    '''

    #convert from BGR to RGB format
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)

    ih,iw = target_size#目标图像大小
    h,w = image.shape[:2]#输入图像的尺寸

    # 缩放比例，
    # 图像缩放时，并非直接缩放的目标尺寸，
    # 而是先将源尺寸的最大边长缩放到指定尺寸，然后进行填充，填充到指定尺寸
    scale = min(iw/w,ih/h)
    nw,nh = int(scale*w),int(scale*h)
    image_resized = cv2.resize(image,(nw,nh))#按最长边的比例进行缩放


    image_paded = np.full(shape=[ih,iw,3],fill_value=128.0)#填充后的目标图像
    dw,dh = (iw-nw)//2,(ih-nh)//2#上/下 左/右填充的宽度，注意//2的作用
    image_paded[dh:nh+dh,dw:nw+dw,:] = image_resized#将缩放后的图像赋值给结果图像.类似于opencv的copyTo()函数

    image_paded = image_paded / 255 #归一化

    if gt_boxes is None:
        return image_paded
    else:#对图像缩放后，针对groundtruth box进行等比例缩放与平移
        gt_boxes[:,[0,2]] = gt_boxes[:,[0,2]] *scale +dw
        gt_boxes[:,[1,3]] = gt_boxes[:,[1,3]] *scale +dh
        return image_paded,gt_boxes



def draw_bbox(image,bboxes,classes=read_class_names(cfg.YOLO.CLASSES),show_label=True):
    '''
    将检测出的目标在图像上用矩形框画出，同时标出其标签及置信度
    :param image:需要绘制结果的图像
    :param bboxes: [x_min,y_min,x_max,y_max,prob,cls_id] format coordinates.
    :param classes:类别名称列表
    :param show_label:是否显示物体类别名称
    :return:绘制了矩形框结果的图像，图像格式和输入图像格式一致(输入为BGR，输入也为BGR；输入为RGB，输出也为RGB格式，)
    '''
    num_classes = len(classes)
    image_h,image_w,_ = image.shape
    hsv_tuples = [(1.0*x/num_classes,1.0,1.0) for x in range(num_classes)]
    colors = list(map(lambda x:colorsys.hsv_to_rgb(*x),hsv_tuples))
    colors = list(map(lambda x:(int(x[0]*255),int(x[1]*255),int(x[2]*255)),colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i,bbox in enumerate(bboxes):
        coor = np.array(bbox[:4],dtype=np.int32)#[xmin,ymin,xmax,ymax]
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6*(image_h+image_w)/600)
        c1,c2 = (coor[0],coor[1]),(coor[2],coor[3])#矩形框的左上和右下顶点
        cv2.rectangle(image,c1,c2,bbox_color,bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f'%(classes[class_ind],score)
            textSize = cv2.getTextSize(bbox_mess,0,fontScale,thickness=bbox_thick//2)[0]

            cv2.rectangle(image,c1,(c1[0]+textSize[0],c1[1]-textSize[1]-3),bbox_color,-1)#fill the label rectangle

            cv2.putText(image,bbox_mess,(c1[0],c1[1]-2),cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale,(0,0,0),bbox_thick//2,lineType=cv2.LINE_AA)

    return image


def bboxes_iou(boxes1,boxes2):
    '''
    cal IOU of two boxes
    :param boxes1:
    :param boxes2:
    :return:
    '''
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    #cal the box's area of boxes1 and boxess
    boxes1Area = (boxes1[...,2]-boxes1[...,0])*(boxes1[...,3]-boxes1[...,1])
    boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    #cal Intersection
    left_up = np.maximum(boxes1[...,:2],boxes2[...,:2])
    right_down = np.minimum(boxes1[...,2:],boxes2[...,2:])

    inter_section = np.maximum(right_down-left_up,0.0)
    inter_area = inter_section[...,0] * inter_section[...,1]
    union_area = boxes1Area+boxes2Area-inter_area
    ious = np.maximum(1.0*inter_area/union_area,np.finfo(np.float32).eps)

    return ious


def read_pb_return_tensors(graph,pb_file,return_elements):
    '''

    :param graph:
    :param pb_file:
    :param return_elements:
    :return:
    '''
    with tf.gfile.FastGFile(pb_file,'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        #从导入的图中得到的与return_element中的名称相对应的操作和/或张量对象的列表
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)

    return return_elements


def nms(bboxes,iou_threshold,sigma=0.3,method='nms'):
    '''
    通过nms过滤掉重合的矩形框
    :param bboxes: (xmin,ymin,xmax,ymax,score,class)
    :param iou_threshold:
    :param sigma:
    :param mehod: 'nms' or soft-nms
    :return:
    '''

    assert method in ['nms','soft-nms']

    classes_in_img = list(set(bboxes[:,5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:,5]==cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes)>0:
            max_ind = np.argmax(cls_bboxes[:,4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes=np.concatenate([cls_bboxes[:max_ind],cls_bboxes[max_ind+1:]])
            iou = bboxes_iou(best_bbox[np.newaxis,:4],cls_bboxes[:, :4])
            weight = np.ones((len(iou),),dtype=np.float32)#?

            if method == 'nms':
                iou_mask = iou>iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0*iou**2/sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4]*weight
            score_mask = cls_bboxes[:,4]>0
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def postprocess_boxes(pred_bbox,org_img_shape,input_size,score_threshold):
    '''
    对检测结果进行处理，过滤掉置信度低于阈值、不在图像范围内的矩形框，返回处理后的结果
    :param pred_bbox:
    :param org_img_shape:检测时，输入所用图像大小
    :param input_size:yolo_v3 推断时所采用的图像大小，416X416
    :param score_threshold:置信度阈值
    :return:[xmin,ymin,xmax,ymax,prob,classid]
    '''
    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]#4列数据内容为：Center_x,Center_y,width,height(中心点坐标+宽高)
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio#计算xmin和xmax在输入图像上的坐标
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio#计算ymin和ymax在输入图像上的坐标

    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)










