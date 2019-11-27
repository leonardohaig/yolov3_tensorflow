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
from tensorflow.python.platform import gfile
from core.yolov3 import YOLOV3

pb_file = "/home/liheng/demo_ckpt/yolov3_bdd100k.pb" # 保存的.pb文件路径
ckpt_file = "/home/liheng/demo_ckpt/yolov3_model_53-epoch.ckpt-261500" # 待转换的.ckpt文件路径

# 需要保存的指定的 节点 名称,而非张量名称
# 节点名称 pred_sbbox 指变量作用空间，concat_2来源于 decode 函数tf.concat操作，由于代码中未显式指定该操作的名称，因此给予了默认名称
# 该默认名称可以通过下面的 print(model.pred_sbbox, model.pred_mbbox, model.pred_lbbox) 打印输出进行查看，打印结果为：
# Tensor("pred_sbbox/concat_2:0", shape=(?, ?, ?, 3, 85), dtype=float32) \
# Tensor("pred_mbbox/concat_2:0", shape=(?, ?, ?, 3, 85), dtype=float32) \
# Tensor("pred_lbbox/concat_2:0", shape=(?, ?, ?, 3, 85), dtype=float32)
# 故务必对concat_2的来源有所知晓！
# output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]
output_node_names = ["input/input_data", "pred_res/pred_bboxes"]# 最终结果
output_node_names = ["input/input_data", "pred_res/openvino_pred_bboxes"]

# 定义模型的输入
with tf.name_scope('input'):
    # input_data = tf.placeholder(dtype=tf.float32, name='input_data')
    input_data = tf.placeholder(dtype=tf.float32, shape=[1,320,320,3],name='input_data')
    trainable = tf.convert_to_tensor(False,dtype=tf.bool,name='training')

model = YOLOV3(input_data, trainable=trainable, bUsedForOpenVINo=True)# 恢复模型之前，首先定义一遍网络结构，然后才能把变量的值恢复到网络中,注意此处trainable=False
print(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)
print(model.pred_sbbox, model.pred_mbbox, model.pred_lbbox)
print(model.pred_res_boxes)

sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                            input_graph_def  = sess.graph.as_graph_def(),#取出图定义
                            output_node_names = output_node_names)#需要保存的指定的 节点 名称
                                                    # 只有定义了输出结点,freeze才会把得到
                                                    # 输出结点所必要的结点都保存下来,或者哪些结点可以丢弃

print('%d ops in the final graph.' % len(converted_graph_def.node))  # 得到当前图有几个操作节点
print('The op.name and op.valuse() are below: ')
for op in sess.graph.get_operations():
   print(op.name, op.values())


print('Fix nodes: ')
# fix nodes
for node in converted_graph_def.node:
    if node.op == 'RefSwitch':
        node.op = 'Switch'
        for index in range(len(node.input)):
            if 'moving_' in node.input[index]:
                node.input[index] = node.input[index] + '/read'
    elif node.op == 'AssignSub':
        node.op = 'Sub'
        if 'use_locking' in node.attr: del node.attr['use_locking']
    elif node.op == 'AssignAdd':
        node.op = 'Add'
        if 'use_locking' in node.attr: del node.attr['use_locking']
    elif node.op == 'Assign':
        node.op = 'Identity'
        if 'use_locking' in node.attr: del node.attr['use_locking']
        if 'validate_shape' in node.attr: del node.attr['validate_shape']
        if len(node.input) == 2:
            # input0: ref: Should be from a Variable node. May be uninitialized.
            # input1: value: The value to be assigned to the variable.
            node.input[0] = node.input[1]
            del node.input[1]
    # print(node.op,':',node.attr)


with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())#将图定义转化为字符串形式并且写入.pb文件中



sess = tf.Session()
with gfile.FastGFile(pb_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')  # 导入计算图

# 需要有一个初始化的过程
sess.run(tf.global_variables_initializer())
for op in sess.graph.get_operations():
   print(op.name, op.values())

exit(0)
