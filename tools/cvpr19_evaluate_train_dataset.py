#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

import os
from tqdm import trange
import argparse
import glob
import cv2
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser(description='The tools for check mot result in mot_folder')
parser.add_argument('--version', default='v1', help='version')
parser.add_argument('--generated_folder', default='/media/ssm/data/dataset/CVPR19/test_logs/mot', help='the image folder')
parser.add_argument('--gt_folder', default='/media/ssm/data/dataset/CVPR19/test_logs/mot/converted', help='the image folder')

args = parser.parse_args()

def evaluate(generated_folder, gt_folder):
    # 1. get the sequence name in generated folder
    ge_path_list = glob.glob(os.path.join(generated_folder, "*.txt"))
    sequence_list = [os.path.splitext(os.path.basename(s))[0] for s in ge_path_list]

    # 2. get the corresponding file list
    gt_path_list = [os.path.join(gt_folder, s+".txt") for s in sequence_list]

    # 3. start evaluate



if __name__ == "__main__":
    evaluate(args.generated_folder, args.gt_folder)

