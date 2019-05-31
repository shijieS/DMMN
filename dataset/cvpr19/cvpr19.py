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

cfg = config[config["phase"]]

class SingleVideoParser:
    """
    A parser for single video.
    """
    def __init__(self, mot_file_path, sequence_name, frames_folder):
        # reading the mot_file
        columns = [
            "frame_index", "track_id", "l", "t", "r", "b",
            "visibility", "class_id"
        ]
        converted_data = pd.read_csv(mot_file_path, sep=',', header=None)
        converted_data = converted_data.loc[:, [0, 1, 2, 3, 4, 5, 8, 7]]
        # converted_data = converted_data.reindex(columns=[0, 1, 2, 3, 4, 5, 7, 6])
        converted_data.columns = columns
        converted_data.loc[:, 'r'] += converted_data.loc[:, 'l']
        converted_data.loc[:, 'b'] += converted_data.loc[:, 't']

        mot_data = converted_data.values
        self.max_frame = np.max(mot_data[:, 0]).astype(int) + 1
        self.max_id = np.max(mot_data[:, 1]).astype(int) + 1
        self.ua_data = np.zeros((self.max_frame, self.max_id, 6), dtype=float)
        self.sequence_frames_folder = sequence_name
        self.frame_folder = frames_folder

        mot_data[:, 6] = (mot_data[:, 6] >= 1 - config["train"]["dataset_overlap_thresh"])
        for row in mot_data:
            self.ua_data[row[0].astype(int), row[1].astype(int), :] = row[2:]

        self.selecte_frame_scale = config['frame_max_input_num'] * config['frame_sample_scale']


    def __len__(self):
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
        ua_data = self.ua_data[frame_indexes+1, :]

        # get ids and bbox
        # 16 x 52
        ua_mask = np.sum(np.abs(ua_data), axis=2) > 0
        # 52
        track_mask = np.sum(ua_mask, axis=0) > config['frame_max_input_num'] * config['min_valid_node_rate']
        track_ids = np.arange(ua_data.shape[1])[track_mask]
        if len(track_ids) == 0:
            return [None, None, None, None, None]
        bboxes = ua_data[:, track_mask, :]

        # get frame path
        frame_paths = [os.path.join(self.frame_folder, "{0:06}.jpg".format(i + 1))
                       for i in frame_indexes]
        frames = [cv2.imread(p) for p in frame_paths]

        h, w, _ = frames[0].shape
        frame_scales = np.array([w, h, w, h])

        # get times
        times = (frame_indexes - frame_indexes[0]) / config["video_fps"]

        bboxes[:, :, :4] /= frame_scales
        return [frame_indexes, track_ids, bboxes, frames, times]



class CVPR19TrainDataset(Dataset):
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

        all_list = glob.glob(os.path.join(self.data_folder, "CVPR19-[0-9][0-9]"))

        if sequence_list is not None:
            sequence_file_list = np.loadtxt(sequence_list, dtype=np.str)
            all_list = [f for f in all_list if os.path.basename(f) in sequence_file_list]

        files_path = [os.path.join(f, "gt/gt.txt") for f in all_list]
        files_name = [os.path.basename(f) for f in all_list]

        # load all the mot files
        self.data = []
        t = trange(len(files_name))
        for name, path, _ in zip(files_name, files_path, t):
            t.set_description('reading: {}'.format(name))
            frames_folder = os.path.realpath(os.path.join(path, "../../img1"))
            self.data += [SingleVideoParser(path, name, frames_folder)]

        # some basic information
        self.lens = [len(p) for p in self.data]
        self.ranges = []
        start_index = 0
        for l in self.lens:
            self.ranges += [(start_index, start_index + l)]
            start_index += l
        self.len = np.sum(self.lens)

    def __len__(self):
        return self.len

    def get_parser(self, item):
        for i, (start, end) in enumerate(self.ranges):
            if item >= start and item < end:
                return self.data[i], item-start
        return None

    # @staticmethod
    # def get_parameters(bboxes, times):
    #     """
    #     Get the parameter of boxes.
    #     :param bboxes: (FrameId, TrackId, 4)
    #     :returns: parameters: (TrackId, ParameterData)
    #               motion_possibility: (trackId, possibility)
    #
    #     """
    #     parameters = list()
    #     motion_posibility = list()
    #     frame_num, track_num, _ = bboxes.shape
    #     mm = MotionModel()
    #     for i in range(track_num):
    #         bbs = bboxes[:, i, :]
    #         mask = np.sum(bbs, axis=1) > 0
    #         if sum(mask) / len(mask) < config['min_valid_node_rate']:
    #             parameters += [MotionModel.get_invalid_params()]
    #             motion_posibility += [0.0]
    #         else:
    #             param = mm.fit(bbs[mask, :], times[mask])
    #             parameters += [param]
    #             motion_posibility += [1.0]
    #     return np.stack(parameters, axis=0), motion_posibility


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

    dataset = CVPR19TrainDataset()
    for index in range(0, len(dataset), 32):
        frame_indexes, track_ids, bboxes, frames, times = dataset[index]

        for i, frame in enumerate(frames):
            h, w, _ = frame.shape
            label_map = {v: k for k, v in config["label_map"].items()}

            texts = [label_map[c] for c in bboxes[i, :, -1].astype(int)]
            DrawBoxes.cv_draw_mult_boxes_with_track(frame,
                                                    bboxes[:, :, :4]*np.array([w, h, w, h]),
                                                    i,
                                                    texts=texts,
                                                    exists=bboxes[:, :, -2].astype(int))
            cv2.imshow("result", frame)
            cv2.waitKey(25)
