#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

import os
import os.path
import torch.utils.data as data
import cv2
import numpy as np
import pandas as pd
from config import config
import random
import glob


class UATestDataset:
    def __init__(self,
                 dataset_path=config['dataset_path'],
                 max_input_frame=config['frame_max_input_num'],
                 frame_scale=config['frame_sample_scale'],
                 sequence_list=config["test"]['sequence_list'],
                 dataset_name='UA-DETRAC'
                 ):
        self.dataset_name = dataset_name
        self.frame_scale = frame_scale
        self.max_frame_num = max_input_frame
        self.max_frame_num_with_scale = max_input_frame * frame_scale
        self.dataset_path = dataset_path

        # 1. get the image folder
        if config["test"]["dataset_type"] == "test":
            folder_name = "Insight-MVT_Annotation_Test"
        else:
            folder_name = "Insight-MVT_Annotation_Train"

        folder = os.path.join(self.dataset_path, folder_name)
        if sequence_list is None:
            image_folder_list = glob.glob(os.path.join(folder, 'MVI_[0-9][0-9][0-9][0-9][0-9]'))
            sequence_list = [os.path.basename(i) for i in image_folder_list]
        else:
            sequence_list = np.loadtxt(sequence_list, dtype=np.str, ndmin=1)
            image_folder_list = [os.path.join(folder, s) for s in sequence_list]

        self.sequence_list = sequence_list

        # 2. get all the image files and its corresponding time
        image_list = [
            sorted(glob.glob(os.path.join(image_folder, 'img[0-9][0-9][0-9][0-9][0-9].jpg')))
            for image_folder in image_folder_list
        ]
        time_list = [
            [
                int(os.path.splitext(os.path.basename(i))[0][3:])
                for i in images_seq
            ]
            for images_seq in image_list
        ]

        # 3. check the time continuity
        for i, time_seq in enumerate(time_list):
            if max(time_seq) - min(time_seq) + 1 != len(time_seq):
                raise AssertionError("sequence {} time not continuity".format(sequence_list[i]))

        # 4. split them into groups which contains self.max_frame frames
        self.all_time_group = []
        self.all_image_group = []
        for time_seq, image_seq in zip(time_list, image_list):
            max_time = max(time_seq)
            min_time = min(time_seq)
            time_group = []
            image_group = []
            for i in range(min_time, max_time - self.max_frame_num_with_scale + 2):
                time_group += [time_seq[i-min_time:i + self.max_frame_num_with_scale - min_time]]
                image_group += [image_seq[i-min_time:i + self.max_frame_num_with_scale - min_time]]

            self.all_time_group += [time_group]
            self.all_image_group += [image_group]

        self.frame_num_list = [len(time_seq) for time_seq in self.all_time_group]

        self.frame_num_range=[]
        start_index = 0
        for l in self.frame_num_list:
            self.frame_num_range += [(start_index, start_index + l)]
            start_index += l

    def __len__(self):
        self.frame_num_range[-1][1]

    def __getitem__(self, item):

        for i, (start, end) in enumerate(self.frame_num_range):
            if item >= start and item < end:
                return [cv2.imread(image_file) for image_file in self.all_image_group[i][item-start]], \
                       (np.array(self.all_time_group[i][item-start]) - min(self.all_time_group[i][item-start])) / \
                       (len(self.all_time_group[i][item-start]) - self.frame_scale)

        return None, None


if __name__ == "__main__":
    dataset = UATestDataset()

    for images, times in dataset:
        for image in images:
            cv2.imshow("result", image)
            cv2.waitKey(30)
        cv2.waitKey(30)
