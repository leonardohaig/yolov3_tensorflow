#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:get_kmeans.py
#       对数据集中的anchor进行聚类
#Date:19-11-5
#Author:liheng
#Version:V1.0
#Reference:https://github.com/lars76/kmeans-anchor-boxes
#============================#

import numpy as np
import cv2


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    param:
        box: tuple or array, shifted to the origin (i. e. width and height)
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = np.true_divide(intersection, box_area + cluster_area - intersection + 1e-10)
    # iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    param:
        boxes: numpy array of shape (r, 4)
    return:
    numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        k: number of clusters
        dist: distance function
    return:
        numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


def parse_anno(annotation_path, target_size=None):
    anno = open(annotation_path, 'r')
    result = []
    for line in anno:
        constent = line.strip().split(' ')
        img_h,img_w,_c = cv2.imread(constent[0],cv2.IMREAD_COLOR).shape
        # img_h, img_w, _c = 720,1280,3
        for item in constent[1:]:
            x_min, y_min, x_max, y_max, class_id = list(map(float,item.split(',')))
            width = x_max - x_min
            height = y_max - y_min
            assert width > 0
            assert height > 0
            # use letterbox resize, i.e. keep the original aspect ratio
            # get k-means anchors on the resized target image size
            if target_size is not None:
                resize_ratio = min(target_size[0] / img_w, target_size[1] / img_h)
                width *= resize_ratio
                height *= resize_ratio
                result.append([width, height])
            # get k-means anchors on the original image size
            else:
                result.append([width, height])

    result = np.asarray(result)
    return result


def get_kmeans(anno, cluster_num=9):

    anchors = kmeans(anno, cluster_num)
    ave_iou = avg_iou(anno, anchors)

    anchors = anchors.astype('int').tolist()

    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    return anchors, ave_iou


if __name__ == '__main__':
    # target resize format: [width, height]
    # if target_resize is speficied, the anchors are on the resized image scale
    # if target_resize is set to None, the anchors are on the original image scale
    target_size = [320, 320]
    annotation_path = "tools/bdd100k_train.txt"
    anno_result = parse_anno(annotation_path, target_size=target_size)
    anchors, ave_iou = get_kmeans(anno_result, 9)

    basline_anchors = []
    strides = [8,8,8,16,16,16,32,32,32]
    basline_anchors_string = ''
    anchor_string = ''
    for i, anchor in enumerate(anchors):
        basline_anchor = [anchor[0]/strides[i], anchor[1]/strides[i]]
        basline_anchors.append(basline_anchor)
        basline_anchors_string += '{},{}, '.format(basline_anchor[0], basline_anchor[1])
        anchor_string += '{},{}, '.format(anchor[0], anchor[1])
    anchor_string = anchor_string[:-2]
    basline_anchors_string = basline_anchors_string[:-2]

    print('anchors are:')
    print(anchor_string)
    print('basline anchors are:')
    print(basline_anchors_string)
    print('the average iou is:')
    print(ave_iou)
