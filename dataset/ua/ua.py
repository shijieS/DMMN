#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

import os
import numpy as np
from config import config
import cv2
from torch.utils.data import Dataset
from tqdm import trange
import random
import glob
import pandas as pd
import xml.etree.ElementTree as ET

cfg = config[config["phase"]]

class SingleVideoParser:
    """
    A parser for single video.
    """
    def __init__(self, mot_file_path, sequence_name, frames_folder):
        # reading the mot_file
        xtree = ET.parse(mot_file_path)
        xroot = xtree.getroot()

        columns = [
            "frame_index", "track_id", "l", "t", "r", "b",
            "overlap_ratio", "object_type"
        ]
        converted_data = []
        for i in range(2, len(xroot)):
            object_num = int(xroot[i].attrib['density'])
            frame_index = int(xroot[i].attrib['num'])
            node = xroot[i][0]
            for j in range(object_num):
                track_id = int(node[j].attrib["id"])
                l = float(node[j][0].attrib["left"])
                t = float(node[j][0].attrib["top"])
                w = float(node[j][0].attrib["width"])
                h = float(node[j][0].attrib["height"])
                r = l + w
                b = t + h
                object_type = node[j][1].attrib["vehicle_type"]
                overlap_ratio = float(node[j][1].attrib["truncation_ratio"])

                converted_data += [[
                    frame_index, track_id, l, t, r, b,
                    overlap_ratio, object_type
                ]]

        converted_data = pd.DataFrame(converted_data, columns=columns)
        converted_data = converted_data.replace({"object_type":config["label_map"]})

        mot_data = converted_data.values
        self.max_frame = int(np.max(mot_data[:, 0])) + 1
        self.max_id = int(np.max(mot_data[:, 1])) + 1
        self.ua_data = np.zeros((self.max_frame, self.max_id, 6), dtype=float)
        self.sequence_name = sequence_name
        self.frames_folder = frames_folder
        self.sequence_frames_folder = os.path.join(self.frames_folder, self.sequence_name)

        image_wh = np.array([config["image_width"], config["image_height"], config["image_width"], config["image_height"]])
        mot_data[:, 2:6] /= image_wh
        mot_data[:, 6] = (mot_data[:, 6] < config["train"]["dataset_overlap_thresh"])
        for row in mot_data:
            self.ua_data[int(row[0])-1, int(row[1])-1, :] = row[2:]

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
        ua_data = self.ua_data[frame_indexes, :]

        # get ids and bbox
        # 16 x 52
        ua_mask = np.sum(np.abs(ua_data), axis=2) > 0
        # 52
        track_mask = np.sum(ua_mask, axis=0) > config['frame_max_input_num'] * config['min_valid_node_rate']
        track_ids = np.arange(ua_data.shape[1])[track_mask]
        if len(track_ids) == 0:
            return [None, None, None, None, None]
        bboxes = ua_data[:, track_mask, :]

        # get frame path (as the frame index is 1-based)
        frame_paths = [os.path.join(self.sequence_frames_folder, "img{0:05}.jpg".format(i + 1)) for i in frame_indexes]
        # frames = []
        # for p in frame_paths:
        #     print(p)
        #     frames += [cv2.imread(p)]
        # # print(frame_paths)
        frames = [cv2.imread(p) for p in frame_paths]

        # get times
        times = (frame_indexes - frame_indexes[0]) / config["video_fps"]

        return [frame_indexes, track_ids, bboxes, frames, times]

class UATrainDataset(Dataset):
    """
    UA Train Dataset
    """
    def __init__(self, root=config['dataset_path'],
                 transform=None,
                 sequence_list=cfg["sequence_list"]):
        """
        Init the UA-DETRAC dataset
        :param root: dataset root
        :param transform: the transform function
        :param sequence_list: the selected sequence list from ua (default location is "./dataset/ua/sequence_list_train.json)
        """
        self.save_folder = os.path.join(root, 'DETRAC-Train-Annotations-Training')
        self.mot_folder = os.path.join(root, 'DETRAC-Train-Annotations-MOT')
        self.frames_folder = os.path.join(root, 'Insight-MVT_Annotation_Train')
        self.xml_folder = os.path.join(root, 'DETRAC-Train-Annotations-XML-v3')


        self.transform = transform
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)

        # analysis files in DETRAC-Train-Annotations-MOT
        files_path = glob.glob(os.path.join(self.xml_folder, "MVI_[0-9][0-9][0-9][0-9][0-9]_v3.xml"))

        if sequence_list is not None:
            sequence_file_list = np.loadtxt(sequence_list, dtype=np.str)
            files_path = list(filter(lambda f: os.path.isfile(f) and os.path.splitext(os.path.basename(f))[0][:-3] in sequence_file_list, files_path))
        files_name = [os.path.basename(f)[:9] for f in files_path]

        # load all the mot files
        self.data = []
        t = trange(len(files_name))
        for name, path, _ in zip(files_name, files_path, t):
            t.set_description('reading: {}'.format(name))
            self.data += [SingleVideoParser(path, name, self.frames_folder)]

        # some basic information
        self.lens = [len(p) for p in self.data]
        self.ranges = []
        start_index = 0
        for l in self.lens:
            self.ranges += [(start_index, start_index+l)]
            start_index += l
        self.len = np.sum(self.lens)

    def __len__(self):
        return self.len

    def get_parser(self, item):
        for i, (start, end) in enumerate(self.ranges):
            if item >= start and item < end:
                return self.data[i], item-start
        return None

    @staticmethod
    def get_parameters(bboxes, times):
        """
        Abandom. Get the parameter of boxes,
        :param bboxes: (FrameId, TrackId, 4)
        :returns: parameters: (TrackId, ParameterData)
                  motion_possibility: (trackId, possibility)

        """
        from motion_model import MotionModel
        parameters = list()
        motion_posibility = list()
        frame_num, track_num, _ = bboxes.shape
        mm = MotionModel()
        for i in range(track_num):
            bbs = bboxes[:, i, :]
            mask = np.sum(bbs, axis=1) > 0
            if sum(mask) / len(mask) < config['min_valid_node_rate']:
                parameters += [MotionModel.get_invalid_params()]
                motion_posibility += [0.0]
            else:
                param = mm.fit(bbs[mask, :], times[mask])
                parameters += [param]
                motion_posibility += [1.0]
        return np.stack(parameters, axis=0), motion_posibility

    def __getitem__(self, item):
        """
        Get the dataset item
        :param item: the 0-based index
        """
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

    dataset = UATrainDataset()
    for index in range(0, len(dataset), 32):
        frame_indexes, track_ids, bboxes, frames, times = dataset[index]

        if frame_indexes is None:
            continue

        for i, frame in enumerate(frames):
            h, w, _ = frame.shape
            label_map = {v: k for k, v in config["label_map"].items()}

            texts = []
            for c in bboxes[i, :, -1].astype(int):
                # print(c)
                if c == 0:
                    texts += ['']
                else:
                    texts += [label_map[c]]
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
