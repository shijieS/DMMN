from dataset.ua import UATrainDataset
from dataset.utils.bbox_show import show_bboxes
from dataset.utils.motion_model_quadratic import MotionModelQuadraticPoly as MM
import cv2
import numpy as np
import os
dataset = UATrainDataset()
frame_index = 0
for frames_1, bboxes_1, motion_parameters_1, motion_possibility_1,\
    frames_2, bboxes_2, motion_parameters_2, motion_possibility_2,\
    similarity_matrix in dataset:

    # generate new bboxes
    times = np.arange(len(frames_1))
    new_bboxes_1 = np.stack([MM(p).get_bbox_by_frames(times) for p in motion_parameters_1], axis=1)
    new_bboxes_2 = np.stack([MM(p).get_bbox_by_frames(times) for p in motion_parameters_2], axis=1)

    frame_left = []
    titles_1 = [MM.get_str(p) for p in motion_parameters_1]
    for i, (frame, bboxes, new_bboxes) in enumerate(zip(frames_1, bboxes_1, new_bboxes_1)):
        frame_left += [
            show_bboxes(show_bboxes(frame, bboxes, color=(255, 0, 0), titles=titles_1),
                        new_bboxes, color=(255, 255, 255))
        ]
    frame_left = np.concatenate(frame_left, axis=0)

    frame_right = []
    titles_2 = [MM.get_str(p) for p in motion_parameters_2]
    for i, (frame, bboxes, new_bboxes) in enumerate(zip(frames_2, bboxes_2, new_bboxes_2)):
        frame_right += [show_bboxes(show_bboxes(frame, bboxes, color=(255, 0, 0), titles=titles_2),
                        new_bboxes, color=(255, 255, 255))
                        ]

    frame_right = np.concatenate(frame_right, axis=0)

    result = np.concatenate([frame_left, frame_right], axis=1)
    cv2.imwrite(os.path.join('/media/jianliu/Data/ssj/github/SSDT/images/', str(frame_index))+'.png',
                result)
    cv2.namedWindow('item', cv2.WINDOW_NORMAL)
    cv2.imshow('item', result)
    frame_index += 1
    cv2.waitKey(20)
