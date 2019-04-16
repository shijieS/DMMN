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
    def __init__(self, mot_file_path, sequence_name, frames_folder):
        # reading the mot_file
        mot_data = np.loadtxt(mot_file_path, delimiter=',')
        self.max_frame = np.max(mot_data[:, 0]).astype(int) + 1
        self.max_id = np.max(mot_data[:, 1]).astype(int) + 1
        self.ua_data = np.zeros((self.max_frame, self.max_id, 4), dtype=float)
        self.sequence_name = sequence_name
        self.frames_folder = frames_folder
        self.sequence_frames_folder = os.path.join(self.frames_folder, self.sequence_name)

        image_wh = np.array([config["image_width"], config["image_height"], config["image_width"], config["image_height"]])
        for row in mot_data:
            self.ua_data[row[0].astype(int), row[1].astype(int), :] = row[2:6] / image_wh

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

        # get frame path
        frame_paths = [os.path.join(self.sequence_frames_folder, "img{0:05}.jpg".format(i + 1)) for i in frame_indexes]
        frames = [cv2.imread(p) for p in frame_paths]

        # get times
        times = (frame_indexes - frame_indexes[0]) / config["video_fps"]

        return [frame_indexes, track_ids, bboxes, frames, times]

class UATrainDataset(Dataset):
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
        self.save_folder = os.path.join(root, 'DETRAC-Train-Annotations-Training')
        self.mot_folder = os.path.join(root, 'DETRAC-Train-Annotations-MOT')
        self.frames_folder = os.path.join(root, 'Insight-MVT_Annotation_Train')
        self.transform = transform
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

    def get_parser(self, item):
        for i, (start, end) in enumerate(self.ranges):
            if item >= start and item < end:
                return self.data[i]
        return None


    def __getitem__(self, item):

        # locate the parser
        parser = self.get_parser(item)
        out = parser[item]
        if parser is None:
            return None

        if self.transform is not None:
            out = self.transform(out)

        return out
