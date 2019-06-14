#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#
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
parser.add_argument('--mot_folder', default='/media/ssm/data/dataset/CVPR19/test_logs/mot', help='the image folder')
parser.add_argument('--save_folder', default='/media/ssm/data/dataset/CVPR19/test_logs/mot/converted', help='the image folder')

args = parser.parse_args()

def check(mot_folder, save_folder):
    sequence_file_list = glob.glob(os.path.join(mot_folder, "*.txt"))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for s in sequence_file_list:
        data = pd.read_csv(s, header=None)
        duplicated_mask = data.duplicated(subset=[0, 1], keep='last')
        duplicated_data = data.loc[duplicated_mask, :]
        if duplicated_data.shape[0] > 0:
            save_data = data.loc[duplicated_mask == False, :].values
            with open(os.path.join(save_folder, os.path.basename(s)), "a+") as f:
                np.savetxt(f, save_data, fmt=["%d", "%d", "%f", "%f", "%f", "%f", "%f", "%d", "%d"], delimiter=",")
            print("There are duplicated row in {}".format(os.path.basename(s)))
            print(duplicated_data)


if __name__ == "__main__":
    check(mot_folder=args.mot_folder, save_folder=args.save_folder)