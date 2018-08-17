import cv2
import numpy as np
import pandas as pd
from config import config
from torch.utils.data import Dataset
import os



class UATrainDataset(Dataset):
    def __init__(self, root = config['dataset_path']):
        pass

    def __getitem__(self, item):
        pre_frame_indexes = None
        pre_frame_bboxes = None
        pre_frame_motion_model = None
        next_frame_indexes = None
        next_frame_bboxes = None
        next_frame_motion_model = None
        similarity_matrix = None


        return pre_frame_indexes, pre_frame_bboxes, pre_frame_motion_model, \
               next_frame_indexes, next_frame_bboxes, next_frame_motion_model, \
               similarity_matrix

