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
from tqdm import trange


class AmotTestDataset:
    def __init__(self,
                 dataset_path=config['dataset_path'],
                 max_input_frame=config['frame_max_input_num'],
                 frame_scale=config['frame_sample_scale'],
                 sequence_list=config["test"]['sequence_list'],
                 dataset_name=config['dataset_name']
                 ):
        self.dataset_name = dataset_name
        self.frame_scale = frame_scale
        self.max_frame_num = max_input_frame
        self.max_frame_num_with_scale = max_input_frame * frame_scale
        self.dataset_path = dataset_path

        # 1. get the image folder
        video_frame_num = 0
        if config["test"]["dataset_type"] == "test":
            folder_name = "test"
            video_frame_num = 3000
        else:
            folder_name = "train"
            video_frame_num = 5000

        folder = os.path.join(self.dataset_path, folder_name)
        if sequence_list is not None:
            sequence_file_list = np.loadtxt(sequence_list, dtype=np.str).reshape((-1))
            self.video_list = [os.path.join(folder, f + ".avi") for f in sequence_file_list]
        else:
            self.video_list = glob.glob(os.path.join(folder, "*/*/*/*.avi"))

        self.video_captures = [cv2.VideoCapture(v) for v in self.video_list]
        # 2. get all the image files and its corresponding time
        time_list = [list(range(video_frame_num)) for _ in self.video_list]

        # 3. check the time continuity
        # for i, time_seq in enumerate(time_list):
        #     if max(time_seq) - min(time_seq) + 1 != len(time_seq):
        #         raise AssertionError("sequence {} time not continuity".format(sequence_list[i]))

        # 4. split them into groups which contains self.max_frame frames
        self.all_time_group = []
        self.video_list_group = []
        for time_seq in time_list:
            max_time = max(time_seq)
            min_time = min(time_seq)
            time_group = []
            for i in range(min_time, max_time - self.max_frame_num_with_scale + 2):
                time_group += [time_seq[i-min_time:i + self.max_frame_num_with_scale - min_time]]

            self.all_time_group += [time_group]

        self.frame_num_list = [len(time_seq) for time_seq in self.all_time_group]

        self.frame_num_range=[]
        start_index = 0
        for l in self.frame_num_list:
            self.frame_num_range += [(start_index, start_index + l)]
            start_index += l


    def get_groupd_index(self, item):
        group_index = -1
        for i, (start, end) in enumerate(self.frame_num_range):
            if item >= start and item < end:
                group_index = i
                break
        return group_index


    def __len__(self):
        return self.frame_num_range[-1][1]

    def get_frame(self, vc_index, item):
        if not self.video_captures[vc_index].isOpened():
            self.video_captures[vc_index].open(self.video_list[vc_index])
        self.video_captures[vc_index].set(cv2.CAP_PROP_POS_FRAMES, item)
        ret, frame = self.video_captures[vc_index].read()
        return frame

    def __getitem__(self, item):

        for i, (start, end) in enumerate(self.frame_num_range):
            if item >= start and item < end:
                return [self.get_frame(i, j) for j in self.all_time_group[i][item-start]], \
                       (np.array(self.all_time_group[i][item-start]) - min(self.all_time_group[i][item-start])) / \
                       (len(self.all_time_group[i][item-start]) - self.frame_scale), \
                        self.all_time_group[i][item - start][0]

        return None, None, None


def get_mean_pixel_value():
    dataset = AmotTestDataset()
    rets = []
    for i in trange(0, len(dataset), 32):
        frame, _, _ = dataset[i]
        if frame is None:
            continue
        a = np.array([0, 0, 0])
        for f in frame:
            a = a+f.sum(axis=0).sum(axis=0)
        b = frame[0].shape[0]*frame[0].shape[1]*len(frame)
        rets += [a / np.array([b, b, b])]


    ret = sum(rets) / len(rets)
    print(ret)
    return ret


if __name__ == "__main__":
    dataset = AmotTestDataset()

    for i in range(0, len(dataset), 32):
        images, times, _ = dataset[i]
        for image in images:
            cv2.imshow("result", image)
            cv2.waitKey(30)
        cv2.waitKey(30)