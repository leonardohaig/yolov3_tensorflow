#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:generate_tfrecord.py
#       生成TFRecord格式的数据（及读取示例）
#Date:19-10-21
#Author:liheng
#Version:V1.0
#============================#

import os
import collections
os.environ["CUDA_VISIBLE_DEVICES"] = "" #不采用GPU
import sys
#sys.path.append("/usr/local/opencv4.1.1/lib/python3.6/dist-packages")
import cv2

import tensorflow as tf

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
    if not isinstance(value, collections.Iterable):
       value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))


def create_tf_example(annotation):

    '''
    创建一条tf_example格式的数据
    :param annotation:list类型，一行label标签，内容：图片路径，目标位置，类别，....
    :return:
    '''

    line = annotation.split()

    image_path = line[0]
    assert os.path.exists(image_path),'{} not exist !'.format(image_path)

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    for content in line[1:]:
        content = list(map(int,content.split(','))) #将其转换为int list
        xmins.append(content[0])
        ymins.append(content[1])
        xmaxs.append(content[2])
        ymaxs.append(content[3])
        classes.append(content[4])

    image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (413, 413), interpolation=cv2.INTER_LINEAR)

    tf_example = tf.train.Example(#key-value形式
        features=tf.train.Features(
            feature={
                'image/image': bytes_feature(image),
                'image/shape': int64_list_feature(list(image.shape)),
                "bbox/xmins": int64_list_feature(xmins),
                "bbox/ymins": int64_list_feature(ymins),
                "bbox/xmaxs": int64_list_feature(xmaxs),
                "bbox/ymaxs": int64_list_feature(ymaxs),
                'image/classes': int64_list_feature(classes),
            }
        ))

    #print(tf_example)

    return tf_example

def generate_tfrecord(labelFile, recordPath):
    '''

    :param labelFile: label file 文件路径
    :param recordPath: 创建的TFRecord文件存储路径
    :return:
    '''

    file_dir = os.path.dirname(os.path.abspath(recordPath))# 获取当前文件所在目录的绝对路径
    assert os.path.exists(file_dir),'{} not exist !'.format(file_dir)

    with open(labelFile,'r') as file:
        # writer = tf.python_io.TFRecordWriter(recordPath)
        writer = tf.io.TFRecordWriter(recordPath)
        for line in file.readlines():
            # annotation = line.split('\n') # 去除末尾的'\n'
            tf_example = create_tf_example(line)
            writer.write(tf_example.SerializeToString())
        writer.close()

    return True


def read_tfrecord(batchsize, recordFileList):
    '''
    从TFRecords文件当中读取图片数据（解析example)
    :param batchsize:
    :param recordFileList: TFRecord file文件列表，list类型
    :return:
    '''

    assert isinstance(recordFileList, collections.Iterable),'param recordFileList need type list!'

    # 1.构造文件队列
    filename_queue = tf.train.string_input_producer(recordFileList,num_epochs=None, shuffle=True)  # 参数为文件名列表

    # 2.构造阅读器
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件

    # 3.批处理,此处批处理提前放置
    batch = tf.train.shuffle_batch([serialized_example],batch_size=batchsize, capacity=batchsize * 5, min_after_dequeue=batchsize * 2,num_threads=1)

    # 4.解析协议块,返回的值是字典.采用tf.parse_example,其返回的Tensor具有batch的维度
    _feature = {'image/image': tf.io.FixedLenFeature([], tf.string),
                'image/shape': tf.io.FixedLenFeature([3], dtype=tf.int64),
                'bbox/xmins': tf.io.VarLenFeature(dtype=tf.int64),
                'bbox/ymins': tf.io.VarLenFeature(dtype=tf.int64),
                'bbox/xmaxs': tf.io.VarLenFeature(dtype=tf.int64),
                'bbox/ymaxs': tf.io.VarLenFeature(dtype=tf.int64),
                'image/classes': tf.io.VarLenFeature(dtype=tf.int64)}
    features = tf.io.parse_example(batch,features=_feature)

    # 得到图片shape信息
    image_shape = features['image/shape']

    # 处理图片数据，由于是一个string,要进行解码，  #将字节转换为数字向量表示，字节为一字符串类型的张量
    # 如果之前用了tostring(),那么必须要用decode_raw()转换为最初的int类型
    # decode_raw()可以将数据从string,bytes转换为int，float类型的
    image_raw = features['image/image']# Get the image as raw bytes.
    image_tensor = tf.decode_raw(image_raw, tf.uint8)# Decode the raw bytes so it becomes a tensor with type.
    # 转换图片的形状，此处需要用动态形状进行转换
    image_tensor = tf.reshape(image_tensor,shape=[batchsize,image_shape[0][0],image_shape[0][1],image_shape[0][2]])
    image_tensor = tf.image.convert_image_dtype(image_tensor,
                                                dtype=tf.float32)  # The type is now uint8 but we need it to be float.

    bbox_xmins = features['bbox/xmins']
    bbox_ymins = features['bbox/ymins']
    bbox_xmaxs = features['bbox/xmaxs']
    bbox_ymaxs = features['bbox/ymaxs']
    bbox_classes = features['image/classes']
    bbox_classes = tf.cast(bbox_classes,dtype=tf.int32)

    bbox_xmins = tf.sparse.to_dense(bbox_xmins)
    bbox_ymins = tf.sparse.to_dense(bbox_ymins)
    bbox_xmaxs = tf.sparse.to_dense(bbox_xmaxs)
    bbox_ymaxs = tf.sparse.to_dense(bbox_ymaxs)
    bbox_classes = tf.sparse.to_dense(bbox_classes)


    return image_tensor,bbox_xmins,bbox_ymins,bbox_xmaxs,bbox_ymaxs,bbox_classes

if __name__ == '__main__':

    # # 生成TFRecords文件
    # generate_tfrecord('/home/liheng/PycharmProjects/yolov3_tensorflow/data/classes/test_yoloTF.txt',
    #                   './test.tfrecord')


    # 从已经存储的TFRecords文件中解析出原始数据
    image_tensor, bbox_xmins, bbox_ymins, bbox_xmaxs, bbox_ymaxs, bbox_classes = read_tfrecord(4,['./test.tfrecord'])
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        # 线程协调器
        coord = tf.train.Coordinator()
        # 开启线程
        thread = tf.train.start_queue_runners(sess, coord)

        for i in range(5):
            _image_tensor, _bbox_xmins, _bbox_ymins, _bbox_xmaxs,\
            _bbox_ymaxs, _bbox_classes = sess.run([image_tensor,bbox_xmins,
                                                   bbox_ymins,bbox_xmaxs,bbox_ymaxs,bbox_classes])


            print(i,_image_tensor.shape)
            #print(_bbox_xmins)

            cv2.imshow('image0', _image_tensor[0])
            cv2.imshow('image1', _image_tensor[1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()




        # 回收线程
        coord.request_stop()
        coord.join(thread)

