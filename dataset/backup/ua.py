import os
import numpy as np
from config import config
import cv2
from torch.utils.data import Dataset
from dataset.MotionModel import MotionModel
from tqdm import tqdm
from tqdm import trange
import random

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
        for row in mot_data:
            self.ua_data[row[0].astype(int), row[1].astype(int), :] = row[2:6]

    def __len__(self):
        return self.max_frame - config['frame_max_input_num']*config['frame_sample_scale']

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
    1. **frames_1** / **frames_2** is the opencv format adjacent frames. Each contains :var:`config['frame_max_input_num']//2` frames.
    2. **bboxes_1** / **bboxes_2** is the bboxes at the first frame.
    3. **motion_parameters_1** / **motion_parameters_1** is the parameters for each bboxes at the first frame.
    4. **motion_possibility_1** / **motion_possibility_1** is the possibility of having corresponding bboxes at the following frames.
    5. **times_1 / times_2 are the 0 based frame indexes.
    5. **similarity_matrix** is the similarity matrix for tracklet in the first frames batch and second frames batch.
    """
    def __init__(self, root=config['dataset_path'], spatial_transform=None, temporal_transform=None, sequence_list=cfg["sequence_list"]):
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
            parser = SingleVideoParser(path)
            for r, i in parser:
                # filter no boxes cases
                if np.sum(i[:len(i)//2, :]) == 0 or np.sum(i[len(i)//2:, :]) == 0:
                    continue
                self.data += [(name, r, i)]

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
        # read one input item
        (video_name, frame_indexes, ua_data) = self.data[item]

        sequence_frames_folder = os.path.join(self.frames_folder, video_name)
        frame_files = [os.path.join(sequence_frames_folder, "img{0:05}.jpg".format(i+1)) for i in frame_indexes]

        total_frame_num = len(frame_indexes)
        range_1 = range(total_frame_num//2)
        range_2 = range(total_frame_num//2, total_frame_num)

        times_1 = frame_indexes[range_1] - frame_indexes[range_1[0]]
        times_2 = frame_indexes[range_2] - frame_indexes[range_2[0]]

        temp_data = np.sum(ua_data, axis=2)
        mask_1 = np.sum(temp_data, axis=0) > 0
        mask_2 = np.sum(temp_data, axis=0) > 0

        # reading the first part of item
        frames_1 = [cv2.imread(frame_files[i]) for i in range_1]
        bboxes_1 = ua_data[range_1, :, :][:, mask_1, :]
        motion_parameters_1, motion_possiblity_1 = MotionModel.get_parameters(
            bboxes_1, times_1,
            config['min_valid_node_rate'])


        # reading the second part of the item
        frames_2 = [cv2.imread(frame_files[i]) for i in range_2]
        bboxes_2 = ua_data[range_2, :, :][:, mask_2, :]
        motion_parameters_2, motion_possiblity_2 = MotionModel.get_parameters(
            bboxes_2, times_2,
            config['min_valid_node_rate'])

        # reading the similarity matrix
        similarity_matrix = np.identity(len(mask_1), dtype=float)[mask_1, :][:, mask_2]

        # generating the classification
        classification_possibility_1 = np.zeros(motion_possiblity_1.shape)

        classification_possibility_2 = np.zeros(motion_possiblity_2.shape)

        out = [frames_1, bboxes_1, motion_parameters_1, motion_possiblity_1, times_1, classification_possibility_1,
               frames_2, bboxes_2, motion_parameters_2, motion_possiblity_2, times_2, classification_possibility_2,
               similarity_matrix]

        if self.temporal_transform is not None:
            out = self.temporal_transform(out)
        if self.spatial_transform is not None:
            out = self.spatial_transform(out)

        return out
