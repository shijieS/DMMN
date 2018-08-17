from ..utils.tracker import Node, Track, Tracks
import os
import numpy as np
from config import config
import pandas as pd

class SingleVideoParser:
    """
    A parser for single video.
    """
    def __init__(self, mot_file_path):
        # reading the mot_file
        datatype = {0: int, 1: int, 2: float, 3: float, 4: float, 5: float}
        mot_data = np.loadtxt(mot_file_path, dtype=datatype, delimiter=',')
        self.max_frame = np.max(mot_data[:, 0])
        self.max_id = np.max(mot_data[:, 1])
        self.ua_data = np.zeros((self.max_frame, self.max_id, 4), dtype=float)
        for row in mot_data:
            self.ua_data[row[0]-1, row[1]-1, :] = mot_data[2:6]

    def __len__(self):
        return self.max_frame - config['frame_max_input_num']

    def __getitem__(self, item):
        return self.ua_data[item:item+config['frame_max_input_num'], :]


class UATrainingDatasetRebuilder:

    def __init__(self, root):
        self.save_folder = os.path.join(root, 'DETRAC-Train-Annotations-Training')
        self.mot_folder = os.path.join(root, 'DETRAC-Train-Annotations-MOT')
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)

        # analysis files in DETRAC-Train-Annotations-MOT
        files_path = [os.path.join(self.mot_folder, f) for f in os.listdir(self.mot_folder)]
        files_path = list(filter(lambda f: os.path.isfile(f), files_path))
        files_name = [os.path.basename(f)[:9] for f in files_path]

        # load all the mot files
        data = {}
        for name, path in zip(files_name, files_path):
            parser = SingleVideoParser(path)
            for i in parser:
                data[name] = i

        # organize them into tracks
        tracks = None

    def run(self):

        pass



