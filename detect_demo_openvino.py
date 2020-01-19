#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:detect_demo_openvino.py
#       利用openvino进行检测
#Date:2019.11.16
#Author:liheng
#Version:V1.0
#============================#

__author__ = 'liheng'
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import time
import logging as log
import core.utils as utils
import numpy as np
import platform

#对于windows平台，执行环境变量设置
# if platform.system() == 'Windows':
if os.name == "nt":
    # import subprocess
    # subprocess.run(
    #     ['call "D:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\bin\setupvars.bat"'])
    cpu_extension = 'E:\LiHeng\Release\cpu_extension_avx2.dll'  # Absolute Path for CPU Extension lib is needed
elif os.name == 'posix':
    cpu_extension = '/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_avx2.so'  # Absolute Path for CPU Extension lib is needed

from openvino.inference_engine import IENetwork, IECore


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=False, type=str, default='/home/liheng/demo_ckpt/FP16/yolov3_my_bdd100k.xml')
    args.add_argument("-i", "--input",
                      help="Required. Path to video file or image. 'cam' for capturing video stream from camera",
                      required=False, type=str,default='/home/liheng/ADAS_Video/1105/ADAS_Video-20191105-154337.mp4')
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                           "kernels implementations.", type=str,
                      default=cpu_extension)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to labels mapping file", default=None, type=str)
    args.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.3, type=float)

    return parser


def main():
    input_size = 320

    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    log.info("Creating Inference Engine...")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    img_info_input_blob = None
    feed_dict = {}
    for blob_name in net.inputs:
        if len(net.inputs[blob_name].shape) == 4:
            input_blob = blob_name
        elif len(net.inputs[blob_name].shape) == 2:
            img_info_input_blob = blob_name
        else:
            raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                               .format(len(net.inputs[blob_name].shape), blob_name))

    assert len(net.outputs) == 1, "Demo supports only single output topologies"

    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
    exec_net = ie.load_network(network=net, num_requests=2, device_name=args.device)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    if img_info_input_blob:
        feed_dict[img_info_input_blob] = [h, w, 1]

    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    cap = cv2.VideoCapture(input_stream)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))  # 此处需要(H,W)格式

    cur_request_id = 0
    next_request_id = 1

    log.info("Starting inference in async mode...")
    is_async_mode = False
    render_time = 0
    ret, frame = cap.read()

    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    print("To switch between sync/async modes, press TAB key in the output window")

    nFrameIdx = 0
    nWaitTime = 1
    while cap.isOpened():
        if is_async_mode:
            ret, next_frame = cap.read()
        else:
            ret, frame = cap.read()
        if not ret:
            break
        nFrameIdx += 1

        initial_w = cap.get(3)
        initial_h = cap.get(4)
        # Main sync point:
        # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
        # in the regular mode we start the CURRENT request and immediately wait for it's completion
        inf_start = time.time()
        if is_async_mode:
            # 图像预处理,转换为RGB格式，并进行缩放
            in_frame = utils.image_preprocess(np.copy(next_frame), [input_size, input_size])
            # in_frame = cv2.cvtColor(in_frame, cv2.COLOR_RGB2BGR).astype(np.float32)

            # in_frame = cv2.resize(next_frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            feed_dict[input_blob] = in_frame
            exec_net.start_async(request_id=next_request_id, inputs=feed_dict)
        else:
            # 图像预处理,转换为RGB格式，并进行缩放
            in_frame = utils.image_preprocess(np.copy(frame), [input_size, input_size])
            # in_frame = cv2.cvtColor(in_frame, cv2.COLOR_RGB2BGR).astype(np.float32)

            # in_frame = cv2.resize(frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            feed_dict[input_blob] = in_frame
            exec_net.start_async(request_id=cur_request_id, inputs=feed_dict)
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            inf_end = time.time()
            det_time = inf_end - inf_start

            # Parse detection results of the current request
            res = exec_net.requests[cur_request_id].outputs[out_blob] # shape[N 6]
            bboxes = utils.postprocess_openvino_boxes(res,frame_size,input_size,[0.57, 0.3])
            bboxes = utils.nms(bboxes, 0.45, method='nms')
            frame = utils.draw_bbox(frame, bboxes)

            curr_time = time.time()
            exec_time = curr_time - inf_start
            info = "Frame:%d Fps: %.2f time: %.2f ms" % (nFrameIdx, 1.0 / exec_time, 1000 * exec_time)

            # Draw performance stats
            inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
                "Inference time: {:.3f} ms".format(det_time * 1000)
            render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1000)
            async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
                "Async mode is off. Processing request {}".format(cur_request_id)

            cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(frame, render_time_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(frame, async_mode_message, (10, int(initial_h - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (10, 10, 200), 1)
            cv2.putText(frame, info, (15, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)

        #
        render_start = time.time()
        cv2.imshow("Detection Results", frame)
        render_end = time.time()
        render_time = render_end - render_start

        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
            frame = next_frame

        key = cv2.waitKey(nWaitTime)
        if key == 27:
            break
        if (9 == key):
            is_async_mode = not is_async_mode
            log.info("Switched to {} mode".format("async" if is_async_mode else "sync"))

        if 32 == key:  # space
            nWaitTime = not nWaitTime


    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)
