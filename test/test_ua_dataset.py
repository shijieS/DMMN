#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

from dataset.ua.ua import UATrainDataset
from dataset.utils.bbox_show import show_bboxes
from motion_model.motion_model_quadratic_backup import MotionModelQuadraticPoly as MM
import cv2
import numpy as np
import os
dataset = UATrainDataset()
frame_index = 0
for frames_1, bboxes_1, motion_parameters_1, p_e_1, times_1, p_c_1, \
    frames_2, bboxes_2, motion_parameters_2, p_e_2, times_2, p_c_2, \
    similarity_matrix in dataset:

    # generate new bboxes
    #times = np.arange(len(frames_1))
    new_bboxes_1 = np.stack([MM(p).get_bbox_by_frames(times_1) for p in motion_parameters_1], axis=1)
    new_bboxes_2 = np.stack([MM(p).get_bbox_by_frames(times_2) for p in motion_parameters_2], axis=1)

    frame_left = []
    titles_1 = [MM.get_str(p) for p in motion_parameters_1]
    for i, (frame, bboxes, new_bboxes, time) in enumerate(zip(frames_1, bboxes_1, new_bboxes_1, times_1)):
        frame = show_bboxes(frame, bboxes, color=(49, 125, 237), titles=titles_1, time=time, alpha=0.15)
        frame = show_bboxes(frame, new_bboxes, color=(71, 173, 112), alpha=0.15)
        frame_left += [frame]
    frame_left = np.concatenate(frame_left, axis=0)

    frame_right = []
    titles_2 = [MM.get_str(p) for p in motion_parameters_2]
    for i, (frame, bboxes, new_bboxes, time) in enumerate(zip(frames_2, bboxes_2, new_bboxes_2, times_2)):
        frame = show_bboxes(frame, bboxes, color=(49, 125, 237), alpha=0.15, titles=titles_1, time=time)
        frame = show_bboxes(frame, new_bboxes, color=(71, 173, 112), alpha=0.15)
        frame_right += [frame]

    frame_right = np.concatenate(frame_right, axis=0)

    result = np.concatenate([frame_left, frame_right], axis=1)
    cv2.imwrite(os.path.join('/home/shiyuan/ssj/logs/ssdt/images-20180925-2', str(frame_index))+'.png',
                result)

    # cv2.namedWindow('item', cv2.WINDOW_NORMAL)
    # cv2.imshow('item', result)
    frame_index += 1
    print(frame_index)
    # cv2.waitKey(20)
