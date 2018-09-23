import torch
from dataset.MotionModel import MotionModel
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from config import config


def generate_targets(org_bboxes, motion_parameters, motion_possibility, times, class_possibility):
    """
    generate a benchmark target
    :param motion_parameters: shape:[N_t
    :param motion_possibility:
    :param times:
    :param class_possibility:
    :return:
    """
    track_num, _, _ = motion_parameters.shape

    # for each track, calculate the corresponding bboxes in each frame
    """
    all_bboxes = []
    for i in range(track_num):
        param = motion_parameters[i, :]
        mm = MotionModel(param.numpy())
        bboxes = mm.get_bbox_by_frames(times.numpy())
        all_bboxes += [bboxes]

    all_bboxes = torch.from_numpy(np.stack(all_bboxes, axis=1))  # (t x N_t x 4)

    # new mask
    mask_predicted = (torch.sum(all_bboxes, dim=2) > 0).double()
    new_mask = (mask_predicted != motion_possibility).double() * mask_predicted

    # 2 means the predict bboxes, 1 means the real bboxes, 0 means the invalid bboxes.
    motion_possibility += new_mask

    org_bboxes[motion_possibility==2.0] = all_bboxes[motion_possibility==2.0]
    """
    # stack the possibility into the bboxes
    target = np.concatenate((org_bboxes,
                             class_possibility[:, :, None],
                             motion_possibility[:, :, None]),
                            axis=2)

    return torch.from_numpy(target).float()


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
        times_1.append(sample[4].float()*0.1)

        frames_2.append(sample[6])
        target_2.append(generate_targets(sample[7], sample[8], sample[9], sample[10], sample[11]))
        times_2.append(sample[10].float()*0.1)

        similarity_matrix.append(sample[12])

    # stack batch
    return torch.stack(frames_1, 0), target_1, torch.stack(times_1, 0), \
           torch.stack(frames_2, 0), target_2, torch.stack(times_2, 0), \
           similarity_matrix

