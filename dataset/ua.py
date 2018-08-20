import os
import numpy as np
from config import config
import cv2
from torch.utils.data import Dataset
from .utils.motion_model_quadratic import MotionModelQuadraticPoly as MotionModel
from tqdm import tqdm
from tqdm import trange

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
        return self.max_frame - config['frame_max_input_num']

    def __getitem__(self, item):
        r = range(item,item+config['frame_max_input_num'])
        return r, self.ua_data[r, :]

class UATrainDataset(Dataset):
    """
    A parser for all videos
    """
    def __init__(self, root=config['dataset_path']):
        self.save_folder = os.path.join(root, 'DETRAC-Train-Annotations-Training')
        self.mot_folder = os.path.join(root, 'DETRAC-Train-Annotations-MOT')
        self.frames_folder = os.path.join(root, 'Insight-MVT_Annotation_Train')
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)

        # analysis files in DETRAC-Train-Annotations-MOT
        files_path = [os.path.join(self.mot_folder, f) for f in os.listdir(self.mot_folder)]
        files_path = list(filter(lambda f: os.path.isfile(f), files_path))
        files_name = [os.path.basename(f)[:9] for f in files_path]

        # load all the mot files
        self.data = []
        t = trange(len(files_name))
        for name, path, _ in zip(files_name, files_path, t):
            t.set_description('reading: {}'.format(name))
            parser = SingleVideoParser(path)
            for r, i in parser:
                self.data += [(name, r, i)]

    def __len__(self):
        return len(self.data)

    def _get_parameters(self, bboxes):
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
                times = np.arange(frame_num)
                param = mm.fit(bbs[mask, :], times[mask])
                parameters += [param]
                motion_posibility += [1.0]
        return np.stack(parameters, axis=0), motion_posibility

    def __getitem__(self, item):
        (video_name, frame_indexes, ua_data) = self.data[item]
        sequence_frames_folder = os.path.join(self.frames_folder, video_name)
        frame_files = [os.path.join(sequence_frames_folder, "img{0:05}.jpg".format(i+1)) for i in frame_indexes]

        total_frame_num = len(frame_indexes)
        range_1 = range(total_frame_num//2)
        range_2 = range(total_frame_num//2, total_frame_num)

        temp_data = np.sum(ua_data, axis=2)
        mask_1 = temp_data[range_1[0], :] > 0
        mask_2 = temp_data[range_2[0], :] > 0


        # reading the first part of item
        frames_1 = [cv2.imread(frame_files[i]) for i in range_1]
        bboxes_1 = ua_data[range_1, :, :][:, mask_1, :]
        motion_parameters_1, motion_possiblity_1 = self._get_parameters(bboxes_1)


        # reading the second part of the item
        frames_2 = [cv2.imread(frame_files[i]) for i in range_2]
        bboxes_2 = ua_data[range_2, :, :][:, mask_2, :]
        motion_parameters_2, motion_possiblity_2 = self._get_parameters(bboxes_2)

        # reading the similarity matrix
        similarity_matrix = np.identity(len(mask_1), dtype=float)[mask_1, :][:, mask_2]

        return  frames_1, bboxes_1, motion_parameters_1, motion_possiblity_1,\
                frames_2, bboxes_2, motion_parameters_2, motion_parameters_2,\
                similarity_matrix