#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

"""
This script refers to the code in motchallenge-devkit

0   frame_index     1-based
1   track_id        1-based
2   left            image coordinate
3   top             image coordinate
4   width           image coordinate
5   height          image coordinate
6   unkown          -
7   class_id
8   visibility


"ped": 1,
"person_on_vhcl": 2,
"car": 3,
"bicycle": 4,
"mbike": 5,
"non_mot_vhcl": 6,
"static_person": 7,
"distractor": 8,
"occluder": 9,
"occluder_on_grnd": 10,
"occluder_full": 11,
"reflection": 12,
"crowd": 13
"""

import os
import numpy as np
from config import config
import cv2
from torch.utils.data import Dataset
from tqdm import trange
import random
import glob
import pandas as pd
from dataset.amot.videocaptureasync import VideoCaptureAsync

cfg = config[config["phase"]]

class SingleVideoParser:
    """
    A parser for single video.
    """
    def __init__(self, gt_file_path, sequence_name, video_file):
        # reading the mot_file
        columns = [
            "frame_index", "track_id", "l", "t", "r", "b",
            "visibility", "class_id"
        ]
        converted_data = pd.read_csv(gt_file_path, index_col=False)
        converted_data = converted_data.loc[:, ['frame_idx', 'id', 'l', 't', 'r', 'b', 'integrity', 'number_of_wheels']]
        # converted_data = converted_data.loc[:, [0, 1, 2, 3, 4, 5, 8, 7]]
        # converted_data = converted_data.reindex(columns=[0, 1, 2, 3, 4, 5, 7, 6])
        converted_data.columns = columns
        # converted_data.loc[:, 'r'] += converted_data.loc[:, 'l']
        # converted_data.loc[:, 'b'] += converted_data.loc[:, 't']

        # filter ignore classes
        converted_data["class_id"] = 1
        # converted_data["class_id"] = converted_data["class_id"].replace(4.0, 1).astype(int)
        # label_map = {v:config["replace_map"][k] for k, v in config["label_map"].items()}
        # converted_data["class_id"] = converted_data["class_id"].map(label_map)
        # converted_data = converted_data[converted_data["class_id"]>-1]

        mot_data = converted_data.values

        if len(mot_data) == 0:
            self.amot_data = None
            return


        self.max_frame = np.max(mot_data[:, 0]).astype(int) + 1
        self.max_id = np.max(mot_data[:, 1]).astype(int) + 1
        self.amot_data = np.zeros((self.max_frame, self.max_id, 6), dtype=float)
        self.sequence_frames_folder = sequence_name
        self.video_file = video_file
        # self.video_capture = VideoCaptureAsync(video_file)

        mot_data[:, 6] = (mot_data[:, 6] >= 1 - config["train"]["dataset_overlap_thresh"])
        for row in mot_data:
            self.amot_data[row[0].astype(int), row[1].astype(int), :] = row[2:]

        self.selecte_frame_scale = config['frame_max_input_num'] * config['frame_sample_scale']

    def get_frame(self, item):
        # if not self.video_capture.isOpened():
        #     self.video_capture.open(self.video_file)
        # print("start {}".format(item))
        # self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, item)
        # ret, frame = self.video_capture.read()
        # print("end {}".format(item))

        # print("start {} {}".format(self.video_file, item))
        ret, frame = VideoCaptureAsync.get_frame(self.video_file, item)
        # print("end {} {}".format(self.video_file, item))

        return ret, frame

    def __len__(self):
        if self.amot_data is None:
            return 0
        return self.max_frame - self.selecte_frame_scale


    def __getitem__(self, item):
        r = np.arange(item, item + self.selecte_frame_scale)
        frame_mask = np.zeros(len(r), dtype=bool)
        if config['random_select_frame']:
            selected_indexes = sorted(random.sample(range(len(r)), config['frame_max_input_num']))
        else:
            selected_indexes = np.arange(0, config['frame_max_input_num']) * config['frame_sample_scale']
        frame_mask[selected_indexes] = True
        frame_indexes = r[frame_mask]
        # print(self.sequence_name)
        amot_data = self.amot_data[frame_indexes, :]

        # get ids and bbox
        # 16 x 52
        amot_mask = np.sum(np.abs(amot_data), axis=2) > 0
        # 52
        track_mask = np.sum(amot_mask, axis=0) > config['frame_max_input_num'] * config['min_valid_node_rate']
        track_ids = np.arange(amot_data.shape[1])[track_mask]
        # if len(track_ids) == 0:
        #     return [None, None, None, None, None]
        bboxes = amot_data[:, track_mask, :]

        # get frame path
        frames = []
        for i in frame_indexes:
            ret, frame = self.get_frame(i)
            frames += [frame]
            if frame is None:
                return [None, None, None, None, None]

        # frame_paths = [os.path.join(self.video_file, "{0:06}.jpg".format(i + 1))
        #                for i in frame_indexes]
        # frames = [cv2.imread(p) for p in frame_paths]

        h, w, _ = frames[0].shape
        frame_scales = np.array([w, h, w, h])

        # get times
        times = (frame_indexes - frame_indexes[0]) / config["video_fps"]

        bboxes[:, :, :4] /= frame_scales
        return [frame_indexes, track_ids, bboxes, frames, times]



class AmotTrainDataset(Dataset):
    def __init__(self, root=config['dataset_path'],
                 transform=None,
                 sequence_list=cfg["sequence_list"]):
        """
        Init the UA-DETRAC dataset
        :param root: dataset root
        :param transform: the spatial transform function
        :param temporal_transform: the temporal transform function
        :param sequence_list: the selected sequence list from ua
        """

        self.data_folder = os.path.join(root, "train")
        self.transform = transform

        if sequence_list is not None:
            sequence_file_list = np.loadtxt(sequence_list, dtype=np.str).reshape((-1))
            all_list = [os.path.join(self.data_folder, f+".avi") for f in sequence_file_list]
        else:
            all_list = glob.glob(os.path.join(self.data_folder, "*/*/*/*.avi"))

        self.sequence_list = all_list
        # check the video files
        for f in all_list:
            if not os.path.exists(f):
                print("Cannot find video fiels {}".format(f))

        gt_files = [os.path.join(os.path.split(f)[0], os.path.join('gt', os.path.splitext(os.path.split(f)[1])[0]+'.csv')) for f in all_list]
        video_files = [f for f in all_list]

        # check the gt_files
        if f in gt_files:
            if not os.path.exists(f):
                print("Cannot find the gt files {}".format(f))

        # load all the mot files
        self.data = []
        t = trange(len(gt_files))

        for gt_file, video_file, _ in zip(gt_files, video_files, t):
            t.set_description('reading: {}'.format(video_file))
            file_name = '{}/{}/{}/'.format(*tuple(video_file.split('/')[-4:-1]))+os.path.splitext(video_file.split('/')[-1])[0]
            self.data += [SingleVideoParser(gt_file, file_name, video_file)]

        # some basic information
        self.lens = [len(p) for p in self.data]
        self.ranges = []
        start_index = 0
        for l in self.lens:
            self.ranges += [(start_index, start_index + l)]
            start_index += l
        self.len = np.sum(self.lens)

    def get_mean_pixel(self):
        rets = []
        for index in trange(0, len(self), 32):
            _, _, _, frames, _ = dataset[index]
            if frames is None:
                continue
            mean_pixel = np.array([0, 0, 0])
            for f in frames:
                mean_pixel = mean_pixel + f.sum(axis=0).sum(axis=0)
            b = f.shape[0] * f.shape[1] * len(frames)
            rets += [mean_pixel / np.array([b, b, b])]
        ret = sum(rets) / len(rets)
        print(ret)
        return ret



    def __len__(self):
        return self.len

    def get_parser(self, item):
        for i, (start, end) in enumerate(self.ranges):
            if item >= start and item < end:
                return self.data[i], item-start
        return None


    def __getitem__(self, item):

        # locate the parser
        parser, item = self.get_parser(item)
        out = parser[item]
        if parser is None:
            return None

        if self.transform is not None:
            out = self.transform(out)

        return out


if __name__ == "__main__":
    from draw_utils.DrawBoxes import DrawBoxes

    dataset = AmotTrainDataset()
    # dataset.get_mean_pixel()
    for index in range(544, len(dataset), 32):
        frame_indexes, track_ids, bboxes, frames, times = dataset[index]

        if frame_indexes is None:
            continue

        for i, frame in enumerate(frames):
            h, w, _ = frame.shape
            label_map = {v: k for k, v in config["label_map"].items()}

            texts = ['' if c==0 else label_map[c] for c in bboxes[i, :, -1].astype(int)]
            colors = []
            for t in track_ids:
                colors += [DrawBoxes.get_random_color(t)]
            DrawBoxes.cv_draw_mult_boxes_with_track(frame,
                                                    bboxes[:, :, :4]*np.array([w, h, w, h]),
                                                    i,
                                                    colors=colors,
                                                    texts=texts,
                                                    exists=bboxes[:, :, -2].astype(int))
            cv2.imshow("result", frame)
            cv2.waitKey(25)
