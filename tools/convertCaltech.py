#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:convertCaltech.py
#       加州理工行人数据集转换脚本
#Date:20-1-13
#Author:liheng
#Version:V1.0
#============================#

import os, glob, argparse
import cv2
from scipy.io import loadmat
from collections import defaultdict
import numpy as np


def vbb_anno2dict(vbb_file, cam_id, person_types=None):
    """
    Parse caltech vbb annotation file to dict
    Args:
        vbb_file: input vbb file path
        cam_id: camera id
        person_types: list of person type that will be used (total 4 types: person, person-fa, person?, people).
            If None, all will be used:
    Return:
        Annotation info dict with filename as key and anno info as value
    """
    filename = os.path.splitext(os.path.basename(vbb_file))[0]
    annos = defaultdict(dict)
    vbb = loadmat(vbb_file)
    # object info in each frame: id, pos, occlusion, lock, posv
    objLists = vbb['A'][0][0][1][0]
    objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
    # person index
    if not person_types:
        person_types = ["person", "person-fa", "person?", "people"]
    person_index_list = [x for x in range(len(objLbl)) if objLbl[x] in person_types]
    for frame_id, obj in enumerate(objLists):
        if len(obj) > 0:
            frame_name = str(cam_id) + "_" + str(filename) + "_" + str(frame_id+1) + ".jpg"
            annos[frame_name] = defaultdict(list)
            annos[frame_name]["id"] = frame_name
            for fid, pos, occl in zip(obj['id'][0], obj['pos'][0], obj['occl'][0]):
                fid = int(fid[0][0]) - 1  # for matlab start from 1 not 0
                if not fid in person_index_list:  # only use bbox whose label is given person type
                    continue
                annos[frame_name]["label"] = objLbl[fid]
                pos = pos[0].tolist()
                occl = int(occl[0][0])
                annos[frame_name]["occlusion"].append(occl)
                annos[frame_name]["bbox"].append(pos)
            if not annos[frame_name]["bbox"]:
                del annos[frame_name]
    return annos



def main():
    vbb_anno2dict('/home/liheng/Downloads/Caltech/annotations/set00/V000.vbb','V000')
    print('hello world !')

if __name__ == '__main__':
    main()