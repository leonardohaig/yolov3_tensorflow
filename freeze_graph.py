#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:freeze_graph.py
#Date:2019.08.14
#Author:liheng
#Version:V1.0
#============================#

__author__ = 'liheng'

import tensorflow as tf
from core.yolov3 import YOLOV3

pb_file = "./yolov3_coco.pb" # 保存的.pb文件路径
ckpt_file = "./checkpoint/yolov3_coco_demo.ckpt" # 待转换的.ckpt文件路径

# 需要保存的指定的 节点 名称,而非张量名称
# 节点名称 pred_sbbox 指变量作用空间，concat_2来源于 decode 函数tf.concat操作，由于代码中未显式指定该操作的名称，因此给予了默认名称
# 该默认名称可以通过下面的 print(model.pred_sbbox, model.pred_mbbox, model.pred_lbbox) 打印输出进行查看，打印结果为：
# Tensor("pred_sbbox/concat_2:0", shape=(?, ?, ?, 3, 85), dtype=float32) \
# Tensor("pred_mbbox/concat_2:0", shape=(?, ?, ?, 3, 85), dtype=float32) \
# Tensor("pred_lbbox/concat_2:0", shape=(?, ?, ?, 3, 85), dtype=float32)
# 故务必对concat_2的来源有所知晓！
output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]

# 定义模型的输入
with tf.name_scope('input'):
    input_data = tf.placeholder(dtype=tf.float32, name='input_data')

model = YOLOV3(input_data, trainable=False)# 恢复模型之前，首先定义一遍网络结构，然后才能把变量的值恢复到网络中,注意此处trainable=False
print(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)
print(model.pred_sbbox, model.pred_mbbox, model.pred_lbbox)

sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                            input_graph_def  = sess.graph.as_graph_def(),#取出图定义
                            output_node_names = output_node_names)#需要保存的指定的 节点 名称
                                                    # 只有定义了输出结点,freeze才会把得到
                                                    # 输出结点所必要的结点都保存下来,或者哪些结点可以丢弃

with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())#将图定义转化为字符串形式并且写入.pb文件中