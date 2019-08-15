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
from itertools import product as product
from layers.ssdt.utils.box_utils import jaccard, point_form
from math import ceil
import torch
from config import config

parser = argparse.ArgumentParser(description='The tools for summary the distribution boxes of UA-DETRAC')
parser.add_argument('--version', default='v1', help='version')
parser.add_argument('--sequence_list', default=config['train']['sequence_list'], help='the sequence list')
parser.add_argument('--dataset_root', default=config['dataset_path'], help='the dataset root')
parser.add_argument('--save_file', default='./ua_data_bboxes.csv', help='the dataset root')

args = parser.parse_args()


def get_all_bboxes(dataset_root, sequence_list, save_file, image_width=960, image_height=540, is_save=True):
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
    if is_save:
        np.savetxt(save_file, data, delimiter=',')
    return data


def get_anchor_boxes(feature_maps, steps, min_sizes, max_sizes, aspect_ratios, scales, image_size):
    anchors = []
    for k, f in enumerate(feature_maps):
        for i, j in product(range(f), repeat=2):
            f_k = image_size / steps[k]
            cx = (i+0.5) / f_k
            cy = (j+0.5) / f_k

            all_s_k = []
            all_s_k += [min_sizes[k] / image_size]
            all_s_k += [np.sqrt(all_s_k[0] * (max_sizes[k] / image_size))]
            for s in scales[k]:
                all_s_k += [(min_sizes[k] + (max_sizes[k] - min_sizes[k]) * s) / image_size]

            for l, s in enumerate(all_s_k):
                anchors += [[cx, cy, s, s]]
                if l == 1:
                    for ratio in aspect_ratios[k]:
                        anchors += [[cx, cy, s/ratio, s*ratio]]
                        anchors += [[cx, cy, s*ratio, s/ratio]]

    return np.array(anchors)


def evaluate_anchor_bboxes(bboxes, anchor_boxes, min_overlap=0.5, min_rect_num=3, batch_size=2000):
    # 1. get the iou matrix between bboxes and anchor boxes
    iteration = ceil(bboxes.shape[0] / batch_size)
    all_valid = []
    for i in trange(iteration):
        start = i * batch_size
        end = (i+1) * batch_size

        iou = jaccard(point_form(torch.from_numpy(bboxes[start:end, :]).cuda()),
                      point_form(torch.from_numpy(anchor_boxes[:]).cuda())).cpu().numpy()

        all_valid += [(iou > min_overlap).sum(axis=1)]


    # 2. calcuate the number of iou larger than min_overlap
    all_valid = np.concatenate(all_valid)
    return (all_valid >= min_rect_num).sum() / bboxes.shape[0], (all_valid > 0).sum() / bboxes.shape[0]



def find_best_anchor_configure(bboxes, input_size=168):
    """
    It's really hard to determine the network configure. Here, we provide this function which can calculate
    the efficient of the generated anchor boxes. We can change the min_size and max_size in their definition
    and do the evaluation.
    :param bboxes:  The bounding boxes of the dataset
    :param input_size: The network input frame size
    :return: The overlap ratio of our gerneated anchor boxes.
    """
    # 1. basic configure
    feature_maps = np.array([42, 21, 11, 6, 3, 2])
    steps = np.array([4, 8, 16, 28, 56, 84])
    min_size = np.array([4, 16, 32, 64, 108, 146])
    max_size = np.array([16, 32, 64, 108, 146, 176])
    aspect_ratios = np.array([[1.5, 2], [2, 3], [2, 3],[2], [2], [2]])
    scales = np.array([[1/1.2, 1/1.5, 1/2.0, 1/2.5], [1/1.2, 1/2.0], [1/1.2, 1/2.0], [1/2.0], [], []])

    # 2. get all anchor boxes
    l, t, r, b = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    bboxes = np.stack([(l+r)/2.0, (t+b)/2.0, r-l, b-t], axis=1)

    anchor_boxes = get_anchor_boxes(feature_maps,
                                    steps,
                                    min_sizes=min_size,
                                    max_sizes=max_size,
                                    aspect_ratios=aspect_ratios,
                                    scales=scales,
                                    image_size=input_size)

    print(anchor_boxes.shape[0])
    valid_ratio, all_ratio = evaluate_anchor_bboxes(bboxes, anchor_boxes, min_overlap=0.45)
    # 3. calculate the location overlap
    print(valid_ratio, all_ratio)


if __name__ == "__main__":
    bboxes = get_all_bboxes(args.dataset_root, args.sequence_list, args.save_file)
    find_best_anchor_configure(bboxes)