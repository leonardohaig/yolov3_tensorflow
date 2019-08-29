#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:video_demo.py
#Date:2019.08.14
#Author:liheng
#Version:V1.0
#============================#

__author__ = 'liheng'

import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf

# 张量名称
return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_coco.pb"
video_path      = "/media/liheng/0F521CCC0F521CCC/7.29/ADAS_usb4mm-20190729-173452.avi"
bSaveResult     = False  # 是否保存结果视频
# video_path      = 0
num_classes     = 80 # 检测目标数量
input_size      = 416
graph           = tf.Graph()
return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)


with tf.Session(graph=graph) as sess:
    vid = cv2.VideoCapture(video_path)
    if bSaveResult:
        videoWriter = cv2.VideoWriter(video_path+'_res.avi',
                                      cv2.VideoWriter_fourcc(*'MJPG'),
                                      20,
                                      (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    nWaitTime = 1
    nFrameIdx = 0
    #cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    while True:
        return_value, frame = vid.read()
        nFrameIdx += 1

        if not return_value:
            print("No image!")
            break

        frame_size = frame.shape[:2]
        image_data = utils.image_preprocess(np.copy(frame), [input_size, input_size])#图像预处理,转换为RGB格式，并进行缩放
        image_data = image_data[np.newaxis, ...]#增加一维，shape从(416,416,3)变为(1,416,416,3)

        prev_time = time.time()

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: image_data})
        # pred_sbbox.shape (1,52,52,3,85)
        # pred_mbbox.shape (1,26,26,3,85)
        # pred_lbbox.shape (1,13,13,3,85)

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),#shape:(8112,85)
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),#shape:(2028,85)
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)#shape:(507,85),pred_bbox.shape:(10647,85)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)#
        bboxes = utils.nms(bboxes, 0.45, method='nms')
        image = utils.draw_bbox(frame, bboxes)

        curr_time = time.time()
        exec_time = curr_time - prev_time

        result = np.asarray(image)
        info = "Frame:%d Fps: %.2f time: %.2f ms" %(nFrameIdx,1.0/exec_time,1000*exec_time)
        cv2.putText(result, info, (0,25), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.imshow("result", result)
        if bSaveResult:
            videoWriter.write(result)


        key = cv2.waitKey(nWaitTime)
        if 27==key:# ESC
            break
        elif 32==key:# space
            nWaitTime = not nWaitTime

    cv2.destroyAllWindows()
