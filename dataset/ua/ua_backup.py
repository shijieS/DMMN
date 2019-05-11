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
from motion_model import MotionModel
from tqdm import trange
import random
from numba import jit

cfg = config[config["phase"]]

class SingleVideoParser:
    """
    A parser for single video.
    """
    def __init__(self, mot_file_path):
        # reading the mot_file
        mot_data = np.loadtxt(mot_file_path, delimiter=',')
        self.max_frame = np.max(mot_data[:, 0]).astype(int) + 1
        self.max_id = np.max(mot_data[:, 1]).astype(int) + 1
        self.ua_data = np.zeros((self.max_frame, self.max_id, 4), dtype=float)

        image_wh = np.array([config["image_width"], config["image_height"], config["image_width"], config["image_height"]])
        for row in mot_data:
            self.ua_data[row[0].astype(int), row[1].astype(int), :] = row[2:6] / image_wh

        select_size = config['frame_max_input_num']*config['frame_sample_scale']
        parameters = []
        for i in trange(self.max_frame):
            parameters += [self.get_boxes_parameters(i, select_size)]
        # parameters = [self.get_boxes_parameters(i, select_size) for i in range(self.max_frame)]

        self.parameters = np.stack(parameters, axis=0)
        # valid_parameters_mask = np.abs(self.parameters).sum(axis=3).sum(axis=2) > 0
        # self.possibilities = np.logical_and(np.sum(np.abs(self.ua_data), axis=2) > 0, valid_parameters_mask)
        # valid_parameters_mask = np.abs(self.parameters).sum(axis=3).sum(axis=2) > 0
        self.possibilities = np.sum(np.abs(self.ua_data), axis=2) > 0

    @jit
    def get_boxes_parameters(self, index, select_size):
        # index_start = int(max(0, index - select_size*config["parameter_frame_scale"]))
        # index_end = int(min(index_start + select_size * (1 + config["parameter_frame_scale"]*2), self.max_frame))

        index_start = index
        index_end = int(min(index_start + select_size, self.max_frame))
        boxes = self.ua_data[index_start:index_end, :, :]
        times = np.arange(index_start, index_end) - index
        ret = MotionModel.get_parameters(boxes, times/config["video_fps"], config['min_valid_node_rate'] / 3.0)
        return ret[0]


    def __len__(self):
        return self.max_frame -  2*config['frame_max_input_num']*config['frame_sample_scale']

    def __getitem__(self, item):
        r = np.arange(item, item + config['frame_max_input_num']*config['frame_sample_scale'])

        # selected frames
        frame_mask = np.zeros(len(r), dtype=bool)
        if config['random_select_frame']:
            selected_indexes = sorted(random.sample(range(len(r)), config['frame_max_input_num']))
        else:
            selected_indexes = np.arange(0, config['frame_max_input_num']) * config['frame_sample_scale']
        frame_mask[selected_indexes] = True
        frame_indexes = r[frame_mask]
        ua_data = self.ua_data[r, :][frame_mask, :]

        return frame_indexes, ua_data

class UATrainDataset(Dataset):
    """ UA Training Dataset Loader
    This is a class for ua training dataset loading.
    The :meth:`__init__` parameters is the ua dataset folder.
    Besides, the return of :meth:`__getitem__` contains 11 items, as following:

    - **frames_1** / **frames_2** is the opencv format adjacent frames. Each contains :var:`config['frame_max_input_num']//2` frames.
    - **bboxes_1** / **bboxes_2** is the bboxes at the first frame.
    - **motion_parameters_1** / **motion_parameters_1** is the parameters for each bboxes at the first frame.
    - **p_e_1** / **p_e_1** is the possibility of having corresponding bboxes at the following frames.
    - **times_1 / times_2 are the 0 based frame indexes.
    - **similarity_matrix** is the similarity matrix for tracklet in the first frames batch and second frames batch.
    """
    def __init__(self, root=config['dataset_path'],
                 spatial_transform=None,
                 temporal_transform=None,
                 sequence_list=cfg["sequence_list"]):
        """
        Init the UA-DETRAC dataset
        :param root: dataset root
        :param spatial_transform: the spatial transform function
        :param temporal_transform: the temporal transform function
        :param sequence_list: the selected sequence list from ua
        """
        self.save_folder = os.path.join(root, 'DETRAC-Train-Annotations-Training')
        self.mot_folder = os.path.join(root, 'DETRAC-Train-Annotations-MOT')
        self.frames_folder = os.path.join(root, 'Insight-MVT_Annotation_Train')
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)

        # analysis files in DETRAC-Train-Annotations-MOT

        files_path = [os.path.join(self.mot_folder, f) for f in os.listdir(self.mot_folder)]
        if sequence_list is None:
            files_path = list(filter(lambda f: os.path.isfile(f), files_path))
        else:
            sequence_file_list = np.loadtxt(sequence_list, dtype=np.str)

            files_path = list(filter(lambda f: os.path.isfile(f) and os.path.splitext(os.path.basename(f))[0] in sequence_file_list, files_path))
        files_name = [os.path.basename(f)[:9] for f in files_path]

        # load all the mot files
        self.data = []
        t = trange(len(files_name))
        for name, path, _ in zip(files_name, files_path, t):
            t.set_description('reading: {}'.format(name))
            parser = SingleVideoParser(path, name, self.frames_folder)
            for frame_indexes, ua_data in parser:
                # filter no boxes cases

                # if np.sum(ua_data[:len(ua_data)//2, :]) == 0 or np.sum(ua_data[len(ua_data)//2:, :]) == 0:
                #     continue

                total_frame_num = len(frame_indexes)
                half_frame_num = total_frame_num // 2
                range_1 = range(half_frame_num)
                range_2 = range(half_frame_num, total_frame_num)

                sequence_frames_folder = os.path.join(self.frames_folder, name)
                frames_1 = frame_indexes[range_1]
                frames_2 = frame_indexes[range_2]
                frames_name_1 = [os.path.join(sequence_frames_folder, "img{0:05}.jpg".format(i + 1)) for i in frames_1]
                frames_name_2 = [os.path.join(sequence_frames_folder, "img{0:05}.jpg".format(i + 1)) for i in frames_2]

                times_1 = (frame_indexes[range_1] - frame_indexes[range_1[0]])/config["video_fps"]
                times_2 = (frame_indexes[range_2] - frame_indexes[range_2[0]])/config["video_fps"]

                mask_1 = np.sum(parser.possibilities[frames_1, :] > 0, axis=0) > half_frame_num * config[
                    'min_valid_node_rate']
                mask_2 = np.sum(parser.possibilities[frames_2, :] > 0, axis=0) > half_frame_num * config[
                    'min_valid_node_rate']

                if np.sum(mask_1) == 0 or np.sum(mask_2) == 0:
                    continue

                bboxes_1 = ua_data[range_1, :, :][:, mask_1, :]
                bboxes_2 = ua_data[range_2, :, :][:, mask_2, :]

                motion_parameters_1 = parser.parameters[frames_1[0], mask_1, :, :]
                motion_parameters_2 = parser.parameters[frames_2[0], mask_2, :, :]

                p_e_1 = parser.possibilities[frames_1, :][:, mask_1]
                p_e_2 = parser.possibilities[frames_2, :][:, mask_2]

                p_c_1 = np.zeros(p_e_1.shape[1])
                p_c_2 = np.zeros(p_e_2.shape[1])

                # reading the similarity matrix
                similarity_matrix = np.identity(len(mask_1), dtype=float)[mask_1, :][:, mask_2]
                extra_row = np.sum(similarity_matrix, axis=1) == 0
                similarity_matrix = np.concatenate((similarity_matrix, extra_row[:, None]), axis=1)
                extra_col = np.sum(similarity_matrix, axis=0) == 0
                similarity_matrix = np.concatenate((similarity_matrix, extra_col[None, :]), axis=0)

                self.data += [(frames_name_1, bboxes_1, motion_parameters_1, p_e_1, times_1, p_c_1,
               frames_name_2, bboxes_2, motion_parameters_2, p_e_2, times_2, p_c_2,
               similarity_matrix)]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_parameters(bboxes, times):
        """
        Get the parameter of boxes.
        :param bboxes: (FrameId, TrackId, 4)
        :returns: parameters: (TrackId, ParameterData)
                  motion_possibility: (trackId, possibility)

        """
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
        Get an item of this dataset
        :param item: the item's index

        **returns:**

        - frames_1 the previous set of frame whose shape is (height, width, channels)
        - bboxes_1: a set of boxes for each frame, whose shape is (frame_num, boxes_num, 4), where 4 means the (l, t, r, b)
        - motion_parameters_1: motion parameters for each boxes whose shape is (boxes_num, 4, 3)
        - p_e_1: the possibility of existing for each boxes. its shape is (frame_num, boxes_num)
        - times_1: the times for each frame. its shape is (frame_num, )
        - p_c_1: the possiblity of being a boxes. its shape is (boxes_num, )
        - frames_2: see frame_1
        - bboxes_2: see bboxes_1
        - motion_parameters_2: see motion_parameters_1
        - p_e_2: see p_e_2
        - times_2: see times_2
        - p_c_2: see p_c_2
        - similarity_matrix: the similarity between boxes in frames_1 and boxes in frames_2. its shape is (boxes_num_in_frames_1+1, boxes_num_in_frames_2+1).
        """
        # read one input item
        (frames_name_1, bboxes_1, motion_parameters_1, p_e_1, times_1, p_c_1,
         frames_name_2, bboxes_2, motion_parameters_2, p_e_2, times_2, p_c_2,
         similarity_matrix) = self.data[item]

        frames_1 = [cv2.imread(n) for n in frames_name_1]
        frames_2 = [cv2.imread(n) for n in frames_name_2]

        # use the parameter to fix the boxes
        motion_bboxes_1 = np.stack([
            MotionModel(motion_parameters_1[i, :]).get_bbox_by_frames(times_1)
            for i in range(motion_parameters_1.shape[0])], axis=1)
        # clean the motion parameters
        p_e_1[np.sum(np.abs(motion_bboxes_1), axis=2) == 0] = False
        zero_mask_1 = np.repeat((np.sum(np.abs(bboxes_1), axis=2) == 0)[:, :, None], repeats=4, axis=2)
        bboxes_1[zero_mask_1] = motion_bboxes_1[zero_mask_1]

        motion_bboxes_2 = np.stack([
            MotionModel(motion_parameters_2[i, :]).get_bbox_by_frames(times_2)
            for i in range(motion_parameters_2.shape[0])], axis=1)
        # clean the motion parameters
        p_e_1[np.sum(np.abs(motion_bboxes_1), axis=2) == 0] = False
        zero_mask_2 = np.repeat((np.sum(np.abs(bboxes_2), axis=2) == 0)[:, :, None], repeats=4, axis=2)
        bboxes_2[zero_mask_2] = motion_bboxes_2[zero_mask_2]

        out = [frames_1, bboxes_1, motion_parameters_1, p_e_1, times_1, p_c_1,
               frames_2, bboxes_2, motion_parameters_2, p_e_2, times_2, p_c_2,
               similarity_matrix]

        if self.temporal_transform is not None:
            out = self.temporal_transform(out)
        if self.spatial_transform is not None:
            out = self.spatial_transform(out)

        return out

        # (video_name, frame_indexes, ua_data, params) = self.data[item]
        #
        # sequence_frames_folder = os.path.join(self.frames_folder, video_name)
        # frame_files = [os.path.join(sequence_frames_folder, "img{0:05}.jpg".format(i+1)) for i in frame_indexes]
        #
        # total_frame_num = len(frame_indexes)
        # half_frame_num = total_frame_num // 2
        # range_1 = range(half_frame_num)
        # range_2 = range(half_frame_num, total_frame_num)
        #
        # times_1 = (frame_indexes[range_1] - frame_indexes[range_1[0]])*0.1
        # times_2 = (frame_indexes[range_2] - frame_indexes[range_2[0]])*0.1
        #
        # temp_data = np.sum(ua_data, axis=2)
        # mask_1 = np.sum(temp_data[:half_frame_num, :] > 0, axis=0) > half_frame_num * config['min_valid_node_rate']
        # mask_2 = np.sum(temp_data[half_frame_num:, :] > 0, axis=0) > half_frame_num * config['min_valid_node_rate']
        #
        # # reading the first part of item
        # frames_1 = [cv2.imread(frame_files[i]) for i in range_1]
        # bboxes_1 = ua_data[range_1, :, :][:, mask_1, :]
        # # p_e_1 is the possibility of existing
        # motion_parameters_1, p_e_1 = MotionModel.get_parameters(
        #     bboxes_1, times_1,
        #     config['min_valid_node_rate'])
        #
        # # # use the parameter to fix the boxes
        # # motion_bboxes_1 = np.stack([
        # #     MotionModel(motion_parameters_1[i, :]).get_bbox_by_frames(times_1)
        # #     for i in range(motion_parameters_1.shape[0])], axis=1)
        # # zero_mask_1 = np.repeat((np.sum(np.abs(bboxes_1), axis=2) == 0)[:, :, None], repeats=4, axis=2)
        # # bboxes_1[zero_mask_1] = motion_bboxes_1[zero_mask_1]
        #
        #
        # # reading the second part of the item
        # frames_2 = [cv2.imread(frame_files[i]) for i in range_2]
        # bboxes_2 = ua_data[range_2, :, :][:, mask_2, :]
        # motion_parameters_2, p_e_2 = MotionModel.get_parameters(
        #     bboxes_2, times_2,
        #     config['min_valid_node_rate'])
        #
        # # reading the similarity matrix
        # similarity_matrix = np.identity(len(mask_1), dtype=float)[mask_1, :][:, mask_2]
        #
        # # generating the classification
        # # p_c_1 is the track label, because there are only 1 label in UA-DETRAC, so p_c_1 is all 0
        # p_c_1 = np.zeros(p_e_1.shape[1])
        #
        # p_c_2 = np.zeros(p_e_2.shape[1])
        #
        # out = [frames_1, bboxes_1, motion_parameters_1, p_e_1, times_1, p_c_1,
        #        frames_2, bboxes_2, motion_parameters_2, p_e_2, times_2, p_c_2,
        #        similarity_matrix]
        #
        # if self.temporal_transform is not None:
        #     out = self.temporal_transform(out)
        # if self.spatial_transform is not None:
        #     out = self.spatial_transform(out)
        #
        # return out
