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
from tqdm import trange


cfg = config[config["phase"]]

class AmotGTDataset(Dataset):
    def __init__(self, root=config['dataset_path'],
                 sequence_list=cfg["sequence_list"]):
        """
        Init the UA-DETRAC dataset
        :param root: dataset root
        :param transform: the spatial transform function
        :param temporal_transform: the temporal transform function
        :param sequence_list: the selected sequence list from ua
        """

        self.data_folder = os.path.join(root, cfg["dataset_type"])

        if sequence_list is not None:
            sequence_file_list = np.loadtxt(sequence_list, dtype=np.str).reshape((-1))
            all_list = [os.path.join(self.data_folder, os.path.split(f)[0] + "/gt/" + os.path.split(f)[1]+".csv") for f in sequence_file_list]
        else:
            all_list = glob.glob(os.path.join(self.data_folder, "*/*/*/*/*.csv"))

        all_tracks_len = 0
        bboxes_len = 0
        frame_len = 0
        columns = [
            "frame_index", "track_id", "l", "t", "r", "b",
            "visibility", "class_id"
        ]
        for _, sequence in zip(trange(len(all_list)), all_list):
            converted_data = pd.read_csv(sequence, index_col=False)
            converted_data = converted_data.loc[:,
                             ['frame_idx', 'id', 'l', 't', 'r', 'b', 'integrity', 'number_of_wheels']]
            converted_data.columns = columns

            all_tracks_len += len(np.unique(converted_data.loc[:, 'track_id'].values))
            bboxes_len += len(converted_data.loc[:, 'track_id'].values)

            if cfg['dataset_type'] == 'train':
                frame_len += 5000
            else:
                frame_len += 3000

        print("========= {} dataset information =========".format(config["phase"]))
        print("frame numbers: {}".format(frame_len))
        print("track numbers: {}".format(all_tracks_len))
        print("boxes numbers: {}".format(bboxes_len))


if __name__ == "__main__":
    dataset = AmotGTDataset()

