import torch
from dataset.MotionModel import MotionModel
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from config import config


def generate_targets(org_bboxes, motion_parameters, p_e, times, p_c):
    """
    generate a benchmark target
    :param motion_parameters: shape:[N_t
    :param p_e:
    :param times:
    :param p_c:
    :return:
    """
    track_num, _, _ = motion_parameters.shape

    # stack the possibility into the bboxes
    target = [
        org_bboxes,
        motion_parameters,
        p_c,
        p_e
    ]

    return target


def collate_fn(batch):
    frames_1 = []               # 0
    target_1 = []               # 1 (N_{fn} x N_{re} x (4+1+1))
    times_1 = []                # 2
    frames_2 = []               # 3
    target_2 = []               # 4 (N_{fn} x N_{re} x (4+1+1))
    times_2 = []                # 5
    similarity_matrix = []      # 6

    # split batch
    for sample in batch:

        # convert to tensor
        frames_1.append(sample[0])
        target_1.append(generate_targets(sample[1], sample[2], sample[3], sample[4], sample[5]))
        times_1.append(sample[4].float())

        frames_2.append(sample[6])
        target_2.append(generate_targets(sample[7], sample[8], sample[9], sample[10], sample[11]))
        times_2.append(sample[10].float())

        similarity_matrix.append(sample[12])

    # stack batch
    return torch.stack(frames_1, 0), target_1, torch.stack(times_1, 0), \
           torch.stack(frames_2, 0), target_2, torch.stack(times_2, 0), \
           similarity_matrix

