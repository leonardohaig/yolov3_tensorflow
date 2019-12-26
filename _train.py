# #!/usr/bin/env python3
# #coding=utf-8
#
# #============================#
# #Program:_train.py
# #       训练模块,尝试对原训练模块进行修改，以增加下列内容：
# #       1.自动加载上次训练结果，恢复训练
# #       2.tf.summary 加入图像展示
# #Date:2019.08.28
# #Author:liheng
# #Version:V1.0
# #Reference:https://github.com/YunYang1994/tensorflow-yolov3/blob/master/train.py
# #============================#
#
# __author__ = 'liheng'
#
# import os
# import time
# import shutil
# import numpy as np
# import tensorflow as tf
# from tqdm import tqdm
#
# import core.utils as utils
# from core.dataset2 import Dataset
# from core.yolov3 import YOLOV3
# from core.config import cfg
#
# # os.environ["CUDA_VISIBLE_DEVICES"] = "" #不采用GPU
# class YoloTrain(object):
#     def __init__(self):
#         self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
#         self.classes = utils.read_class_names(cfg.YOLO.CLASSES)#dict类型，ID---name
#         self.num_classes = len(self.classes)#检测类别数量
#         self.learn_rate_init = cfg.TRAIN.LEARN_RATE_INIT
#         self.learn_rate_end = cfg.TRAIN.LEARN_RATE_END
#         self.first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
#         self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
#         self.warmup_periods = cfg.TRAIN.WARMUP_EPOCHS
#         self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY #滑动平均时的decay值
#
#         self.train_logdir = cfg.TRAIN.LOG_SAVE_DIR  # 训练日志保存路径
#         if os.path.exists(self.train_logdir): shutil.rmtree(self.train_logdir)  # 递归删除文件夹下的所有子文件夹和子文件
#         os.mkdir(self.train_logdir)
#
#         self.config = tf.ConfigProto(allow_soft_placement=True)
#         # self.config.gpu_options.allow_growth = True
#         self.config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 占用80%显存
#         self.sess = tf.Session(config=self.config)
#         self.coord = tf.train.Coordinator()#线程协调器
#         self.ckpt_savePath = cfg.TRAIN.MODEL_SAVE_DIR # ckpt文件保存路径
#         if not os.path.exists(self.ckpt_savePath):  # 模型保存路径不存在，则创建该路径
#             os.mkdir(self.ckpt_savePath)
#
#         self.dataset_type = tf.placeholder(dtype=tf.bool, name='isTrain') # 占位符，对训练时，为True，验证时为False
#         self.trainset = Dataset('train', self.sess, self.coord)
#         self.testset = Dataset('test',self.sess, self.coord)
#         self.steps_per_period = len(self.trainset)
#
#
#         with tf.name_scope('define_input'):
#             def train_fn():
#                 tensor_data = self.trainset.tensor_data
#                 return tensor_data
#             def test_fn():
#                 tensor_data = self.testset.tensor_data
#                 return tensor_data
#             self.tensor_data = tf.cond(pred=self.dataset_type,true_fn=train_fn,false_fn=test_fn)
#
#             # self.input_data = tf.placeholder(dtype=tf.float32, shape=[None,None,None,None], name='input_data')
#             self.input_data = self.tensor_data[0]
#             self.score_threshold = tf.placeholder(dtype=tf.float32, shape=[], name='score_threshold')
#             self.iou_threshold = tf.placeholder(dtype=tf.float32, shape=[], name='iou_threshold')
#             self.per_cls_maxboxes = tf.placeholder(dtype=tf.int32, shape=[], name='per_cls_maxboxes')
#             # self.label_sbbox = tf.placeholder(dtype=tf.float32, name='label_sbbox')
#             self.label_sbbox = self.tensor_data[1]
#             # self.label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')
#             self.label_mbbox = self.tensor_data[2]
#             # self.label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')
#             self.label_lbbox = self.tensor_data[3]
#             # self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
#             self.true_sbboxes = self.tensor_data[4]
#             # self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
#             self.true_mbboxes = self.tensor_data[5]
#             # self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
#             self.true_lbboxes = self.tensor_data[6]
#             # self.input_gt_image = tf.placeholder(dtype=tf.float64, name='input_gt_image')
#             self.input_gt_image = self.tensor_data[7]
#             self.train_pred_image = tf.placeholder(dtype=tf.float64, name='train_pred_image')
#             self.trainable = tf.placeholder(dtype=tf.bool, name='training') # 占位符，对训练时，为True，验证时为False
#
#
#
#
#         with tf.name_scope("define_loss"):
#             self.model = YOLOV3(self.input_data, self.trainable,self.score_threshold,self.iou_threshold,self.per_cls_maxboxes)
#             self.net_var = tf.global_variables()
#             self.giou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(
#                 self.label_sbbox, self.label_mbbox, self.label_lbbox,
#                 self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
#             self.loss = self.giou_loss + self.conf_loss + self.prob_loss # 损失函数
#             self.pred_res_boxes = self.model.get_imgage_predbboxes()
#             self.train_pred_image = self.model.get_pred_image(self.input_data)
#
#         with tf.name_scope('learn_rate'):
#             self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
#             warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
#                                        dtype=tf.float64, name='warmup_steps')
#             train_steps = tf.constant((self.first_stage_epochs + self.second_stage_epochs) * self.steps_per_period,
#                                       dtype=tf.float64, name='train_steps')
#             self.learn_rate = tf.cond(
#                 pred=self.global_step < warmup_steps,
#                 true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
#                 false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
#                                  (1 + tf.cos(
#                                      (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
#             )
#             global_step_update = tf.assign_add(self.global_step, 1.0)
#
#         with tf.name_scope("define_weight_decay"):
#             moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())#给模型中的变量创建滑动平均（滑动平均，作用于模型中的变量）
#
#         # 第一阶段训练：仅仅训练三个分支的最后卷积层
#         with tf.name_scope("define_first_stage_train"):
#             self.first_stage_trainable_var_list = []
#             for var in tf.trainable_variables():
#                 var_name = var.op.name
#                 var_name_mess = str(var_name).split('/')
#                 if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
#                     self.first_stage_trainable_var_list.append(var)
#
#             first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
#                                                                                      var_list=self.first_stage_trainable_var_list)
#             with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
#                 with tf.control_dependencies([first_stage_optimizer, global_step_update]):
#                     with tf.control_dependencies([moving_ave]):
#                         self.train_op_with_frozen_variables = tf.no_op()
#
#         # 第二阶段训练：训练所有的层，其实也就是 fine-tunning 阶段
#         with tf.name_scope("define_second_stage_train"):
#             second_stage_trainable_var_list = tf.trainable_variables()
#             second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
#                                                                                       var_list=second_stage_trainable_var_list)
#
#             with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
#                 with tf.control_dependencies([second_stage_optimizer, global_step_update]):
#                     with tf.control_dependencies([moving_ave]):
#                         self.train_op_with_all_variables = tf.no_op()
#
#         with tf.name_scope('loader_and_saver'):
#             self.loader = tf.train.Saver(self.net_var)
#             self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5) # 仅保留最近5次的结果
#
#         with tf.name_scope('summary'):
#             tf.summary.scalar("learn_rate", self.learn_rate)
#             tf.summary.scalar("giou_loss", self.giou_loss)
#             tf.summary.scalar("conf_loss", self.conf_loss)
#             tf.summary.scalar("prob_loss", self.prob_loss)
#             tf.summary.scalar("total_loss", self.loss)
#             tf.summary.image("input_gt_image", self.input_gt_image, max_outputs=3)
#             tf.summary.image("train_pred_image",self.train_pred_image, max_outputs=3)
#
#             self.write_op = tf.summary.merge_all()#将所有summary全部保存到磁盘,以便tensorboard显示
#             self.summary_writer = tf.summary.FileWriter(self.train_logdir, graph=self.sess.graph)#指定一个文件用来保存图
#
#     def train(self):
#         self.sess.run(tf.global_variables_initializer())
#         # 加载上次已经训练后的权重
#         try:
#             print('=> Restoring weights from last trained file ...')
#             last_checkpoint = tf.train.latest_checkpoint(self.ckpt_savePath)  # 会自动找到最近保存的变量文件
#             self.loader.restore(self.sess, last_checkpoint)
#         except:
#             print('=> Can not find last trained file !!!')
#             print('=> Now it starts to train YOLOV3 from scratch ...')
#             self.first_stage_epochs = 0
#
#
#         print('=> Start train,total epoch is:%d' % (self.first_stage_epochs + self.second_stage_epochs) )
#
#         for epoch in range(1, 1 + self.first_stage_epochs + self.second_stage_epochs):
#             if epoch <= self.first_stage_epochs:
#                 train_op = self.train_op_with_frozen_variables
#             else:
#                 train_op = self.train_op_with_all_variables
#
#             pbar = tqdm(range(self.trainset.__len__()))
#             train_epoch_loss, test_epoch_loss = [], []
#             global_step_val = 0 #当前的训练步数
#
#             for train_data in pbar:
#             # for train_data in range(self.trainset.__len__()):
#                 _, summary, train_step_loss, global_step_val, pred_boxes= self.sess.run(
#                     [train_op, self.write_op, self.loss, self.global_step,self.pred_res_boxes], feed_dict={
#                         self.dataset_type:True,
#                         # self.input_data: train_data[0],
#                         # self.label_sbbox: train_data[1],
#                         # self.label_mbbox: train_data[2],
#                         # self.label_lbbox: train_data[3],
#                         # self.true_sbboxes: train_data[4],
#                         # self.true_mbboxes: train_data[5],
#                         # self.true_lbboxes: train_data[6],
#                         # self.input_gt_image: train_data[7],
#                         self.score_threshold: 0.3,
#                         self.iou_threshold: 0.45,
#                         self.per_cls_maxboxes: 50,
#                         self.trainable: True,
#                     })
#
#                 global_step_val = int(global_step_val)
#                 train_epoch_loss.append(train_step_loss)
#                 self.summary_writer.add_summary(summary, global_step_val)
#                 pbar.set_description("Step:%d train loss: %.2f" % (global_step_val,train_step_loss))
#
#                 #每500step额外保存一个模型
#                 if global_step_val % 500 == 0:
#                     ckpt_file = os.path.join(self.ckpt_savePath, 'yolov3_model_%d-epoch.ckpt' % epoch)
#                     self.saver.save(self.sess, ckpt_file, global_step=global_step_val)
#
#             for test_data in range(self.testset.__len__()):
#                 test_step_loss = self.sess.run(self.loss, feed_dict={
#                     self.dataset_type: False,
#                     # self.input_data: test_data[0],
#                     # self.label_sbbox: test_data[1],
#                     # self.label_mbbox: test_data[2],
#                     # self.label_lbbox: test_data[3],
#                     # self.true_sbboxes: test_data[4],
#                     # self.true_mbboxes: test_data[5],
#                     # self.true_lbboxes: test_data[6],
#                     self.trainable: False,
#                 })
#
#                 test_epoch_loss.append(test_step_loss)
#
#             train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
#
#             ckpt_file = os.path.join(self.ckpt_savePath,'yolov3_test_loss=%.4f_%d-epoch.ckpt' %(test_epoch_loss,epoch))
#             log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
#             print("=> Epoch: %2d/%2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
#                   % (epoch,(self.first_stage_epochs + self.second_stage_epochs),log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
#             self.saver.save(self.sess, ckpt_file, global_step=global_step_val)
#
#         #结束数据读取线程
#         # self.trainset.stopThread()
#         # self.testset.stopThread()
#         self.coord.request_stop()  # 通知其他线程关闭
#         self.coord.join(self.trainset.enqueue_threads)  # join 操作等待其他线程结束,其他所有线程关闭之后,这一函数才能返回
#         self.coord.join(self.testset.enqueue_threads)
#         os._exit(0)
#
#
# if __name__ == '__main__':
#     YoloTrain().train()




# """

#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:_train.py
#       训练模块,尝试对原训练模块进行修改，以增加下列内容：
#       1.自动加载上次训练结果，恢复训练
#       2.tf.summary 加入图像展示
#Date:2019.08.28
#Author:liheng
#Version:V1.0
#Reference:https://github.com/YunYang1994/tensorflow-yolov3/blob/master/train.py
#============================#

__author__ = 'liheng'

import os
import time
import shutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import core.utils as utils
from core.dataset import Dataset
from core.yolov3 import YOLOV3
from core.config import cfg

#os.environ["CUDA_VISIBLE_DEVICES"] = "" #不采用GPU
class YoloTrain(object):
    def __init__(self):
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)#dict类型，ID---name
        self.num_classes = len(self.classes)#检测类别数量
        self.learn_rate_init = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods = cfg.TRAIN.WARMUP_EPOCHS
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY #滑动平均时的decay值

        self.train_logdir = cfg.TRAIN.LOG_SAVE_DIR  # 训练日志保存路径
        if os.path.exists(self.train_logdir): shutil.rmtree(self.train_logdir)  # 递归删除文件夹下的所有子文件夹和子文件
        os.mkdir(self.train_logdir)

        self.trainset = Dataset('train')
        self.testset = Dataset('test')
        self.steps_per_period = len(self.trainset)
        self.config = tf.ConfigProto(allow_soft_placement=True)
        # self.config.gpu_options.allow_growth = True
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 占用80%显存
        self.sess = tf.Session(config=self.config)
        self.ckpt_savePath = cfg.TRAIN.MODEL_SAVE_DIR # ckpt文件保存路径
        if not os.path.exists(self.ckpt_savePath):  # 模型保存路径不存在，则创建该路径
            os.mkdir(self.ckpt_savePath)

        with tf.name_scope('define_input'):
            self.input_data = tf.placeholder(dtype=tf.float32, shape=[None,None,None,None], name='input_data')
            self.score_threshold = tf.placeholder(dtype=tf.float32, shape=[], name='score_threshold')
            self.iou_threshold = tf.placeholder(dtype=tf.float32, shape=[], name='iou_threshold')
            self.per_cls_maxboxes = tf.placeholder(dtype=tf.int32, shape=[], name='per_cls_maxboxes')
            self.label_sbbox = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.input_gt_image = tf.placeholder(dtype=tf.float64, name='input_gt_image')
            self.train_pred_image = tf.placeholder(dtype=tf.float64, name='train_pred_image')
            self.trainable = tf.placeholder(dtype=tf.bool, name='training') # 占位符，对训练时，为True，验证时为False

        with tf.name_scope("define_loss"):
            self.model = YOLOV3(self.input_data, self.trainable,self.score_threshold,self.iou_threshold,self.per_cls_maxboxes)
            self.net_var = tf.global_variables()
            self.giou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(
                self.label_sbbox, self.label_mbbox, self.label_lbbox,
                self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss # 损失函数
            self.pred_res_boxes = self.model.get_imgage_predbboxes()
            self.train_pred_image = self.model.get_pred_image(self.input_data)

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                       dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant((self.first_stage_epochs + self.second_stage_epochs) * self.steps_per_period,
                                      dtype=tf.float64, name='train_steps')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                 (1 + tf.cos(
                                     (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.assign_add(self.global_step, 1.0)

        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())#给模型中的变量创建滑动平均（滑动平均，作用于模型中的变量）

        # 第一阶段训练：仅仅训练三个分支的最后卷积层
        with tf.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                                                     var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        # 第二阶段训练：训练所有的层，其实也就是 fine-tunning 阶段
        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                                                      var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5) # 仅保留最近5次的结果

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate", self.learn_rate)
            tf.summary.scalar("giou_loss", self.giou_loss)
            tf.summary.scalar("conf_loss", self.conf_loss)
            tf.summary.scalar("prob_loss", self.prob_loss)
            tf.summary.scalar("total_loss", self.loss)
            tf.summary.image("input_gt_image", self.input_gt_image, max_outputs=3)
            tf.summary.image("train_pred_image",self.train_pred_image, max_outputs=3)

            self.write_op = tf.summary.merge_all()#将所有summary全部保存到磁盘,以便tensorboard显示
            self.summary_writer = tf.summary.FileWriter(self.train_logdir, graph=self.sess.graph)#指定一个文件用来保存图

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        # 加载上次已经训练后的权重
        try:
            print('=> Restoring weights from last trained file ...')
            last_checkpoint = tf.train.latest_checkpoint(self.ckpt_savePath)  # 会自动找到最近保存的变量文件
            self.loader.restore(self.sess, last_checkpoint)
        except:
            print('=> Can not find last trained file !!!')
            print('=> Now it starts to train YOLOV3 from scratch ...')
            self.first_stage_epochs = 0


        print('=> Start train,total epoch is:%d' % (self.first_stage_epochs + self.second_stage_epochs) )

        for epoch in range(1, 1 + self.first_stage_epochs + self.second_stage_epochs):
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables

            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []
            global_step_val = 0 #当前的训练步数

            for train_data in pbar:
                _, summary, train_step_loss, global_step_val, pred_boxes= self.sess.run(
                    [train_op, self.write_op, self.loss, self.global_step,self.pred_res_boxes], feed_dict={
                        self.input_data: train_data[0],
                        self.label_sbbox: train_data[1],
                        self.label_mbbox: train_data[2],
                        self.label_lbbox: train_data[3],
                        self.true_sbboxes: train_data[4],
                        self.true_mbboxes: train_data[5],
                        self.true_lbboxes: train_data[6],
                        self.input_gt_image: train_data[7],
                        self.score_threshold: 0.3,
                        self.iou_threshold: 0.45,
                        self.per_cls_maxboxes: 50,
                        self.trainable: True,
                    })

                global_step_val = int(global_step_val)
                train_epoch_loss.append(train_step_loss)
                self.summary_writer.add_summary(summary, global_step_val)
                pbar.set_description("Step:%d train loss: %.2f" % (global_step_val,train_step_loss))

                #每500step额外保存一个模型
                if global_step_val % 500 == 0:
                    ckpt_file = os.path.join(self.ckpt_savePath, 'yolov3_model_%d-epoch.ckpt' % epoch)
                    self.saver.save(self.sess, ckpt_file, global_step=global_step_val)

            for test_data in self.testset:
                test_step_loss = self.sess.run(self.loss, feed_dict={
                    self.input_data: test_data[0],
                    self.label_sbbox: test_data[1],
                    self.label_mbbox: test_data[2],
                    self.label_lbbox: test_data[3],
                    self.true_sbboxes: test_data[4],
                    self.true_mbboxes: test_data[5],
                    self.true_lbboxes: test_data[6],
                    self.trainable: False,
                })

                test_epoch_loss.append(test_step_loss)

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)

            ckpt_file = os.path.join(self.ckpt_savePath,'yolov3_test_loss=%.4f_%d-epoch.ckpt' %(test_epoch_loss,epoch))
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d/%2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                  % (epoch,(self.first_stage_epochs + self.second_stage_epochs),log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
            self.saver.save(self.sess, ckpt_file, global_step=global_step_val)

        #结束数据读取线程
        self.trainset.stopThread()
        self.testset.stopThread()


if __name__ == '__main__':
    YoloTrain().train()

# """