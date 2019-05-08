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
from dataset.ua.ua import UATrainDataset
import numpy as np


parser = argparse.ArgumentParser(description='The tools for summary the distribution boxes of UA-DETRAC')
parser.add_argument('--version', default='v1', help='version')
parser.add_argument('--sequence_list', default='../dataset/sequence_list_all.txt', help='the sequence list')
parser.add_argument('--dataset_root', default='/media/ssm/data/dataset/uadetrac/', help='the dataset root')
parser.add_argument('--save_file', default='./ua_data_bboxes.csv', help='the dataset root')

args = parser.parse_args()


def start(dataset_root, sequence_list, save_file, image_width=960, image_height=540):
    dataset = UATrainDataset(root=dataset_root,
                             sequence_list=sequence_list)


    # 1. get all the ground truth file path
    gt_root = os.path.join(dataset_root, 'DETRAC-Train-Annotations-MOT')
    files_path = [os.path.join(gt_root, f) for f in os.listdir(gt_root)]
    sequence_file_list = np.loadtxt(sequence_list, dtype=np.str)
    files_path = list(
        filter(
            lambda f: os.path.isfile(f) and os.path.splitext(os.path.basename(f))[0] in sequence_file_list,
               files_path)
    )

    # 2. select bboxes and connect them together
    all_raw_data = []
    for _, gt_file in zip(trange(len(files_path)), files_path):
        mot_data = np.loadtxt(gt_file, delimiter=',')
        all_raw_data += [mot_data]

    data = np.concatenate(all_raw_data, axis=0)
    data = data[:, 2:]
    data[:, [0, 2]] /= image_width
    data[:, [1, 3]] /= image_height
    np.savetxt(save_file, data[:, 2:], delimiter=',')



if __name__ == "__main__":
    start(args.dataset_root, args.sequence_list, args.save_file)