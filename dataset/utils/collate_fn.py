import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from config import config


def collate_fn(batch):
    frames_1 = []               # 0
    bboxes_1 = []               # 1
    motion_parameters_1 = []    # 2
    motion_possiblity_1 = []    # 3
    times_1 = []                # 4
    frames_2 = []               # 5
    bboxes_2 = []               # 6
    motion_parameters_2 = []    # 7
    motion_possiblity_2 = []    # 8
    times_2 = []                # 9
    similarity_matrix = []      # 10

    # split batch
    for sample in batch:
        frames_1.append(sample[0])
        bboxes_1.append(sample[1])
        motion_parameters_1.append(sample[2])
        motion_possiblity_1.append(sample[3])
        times_1.append(sample[4])

        frames_2.append(sample[5])
        bboxes_2.append(sample[6])
        motion_parameters_2.append(sample[7])
        motion_possiblity_2.append(sample[8])
        times_2.append(sample[9])

        similarity_matrix.append(sample[10])

    # stack batch
    return torch.stack(frames_1, 0), torch.stack(bboxes_1, 0),\
           torch.stack(motion_parameters_1, 0), torch.stack(motion_possiblity_1, 0), \
           torch.stack(times_1, 0), \
           torch.stack(frames_2, 0), torch.stack(bboxes_2, 0), \
           torch.stack(motion_parameters_2, 0), torch.stack(motion_possiblity_2, 0), \
           torch.stack(times_2, 0), \
           torch.stack(similarity_matrix, 0),

