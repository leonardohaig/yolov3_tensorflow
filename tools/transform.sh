#!/usr/bin/env bash
#Pragram:转换.pb文件为openvino格式
#Date:  2020.1.19
#Author:liheng
#Verson:V1.0

python3 /opt/intel/openvino_2019.3.334/deployment_tools/model_optimizer/mo.py \
--input_model /home/liheng/demo_ckpt/yolov3_my_bdd100k.pb \
--output_dir /home/liheng/demo_ckpt/FP16 \
--data_type FP16