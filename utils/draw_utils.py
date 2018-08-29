from config import config
import cv2
import torch
import numpy as np
import datetime
import os


def show_bboxes(frames, targets, is_save=True, iteration=None):
    """
    draw a corresponding rectangles to the frames
    :param frames_1: all the frames
    :param target_1: shape: [N_{ba}, 3, N_{ti}, W, H]
    :param times_1: shape: [N_{ba}, N_{ti}, (4+1+1)], where '4': bbox, '1': class_possibility, '1': motion_possibility
    :return: a list for each batch which contains all the frames
    """
    N_batch, _, N_time, W, H = frames.shape
    result = []
    for n in range(N_batch):
        result_frames = []
        for t in range(N_time):
            frame = frames[n, :, t].cpu().numpy()
            frame = frame.transpose([1, 2, 0]) + config['pixel_mean']
            frame = np.clip(frame, 0, 255).astype(np.uint8).copy()

            target = targets[n][t, :].cpu().numpy()

            for bbox in target:
                if bbox[-1] != 0:
                    bbox[:4] *= config["frame_size"]
                    frame = cv2.rectangle(frame, tuple(bbox[:2].astype(int)), tuple(bbox[2:4].astype(int)), (255, 0, 0))
            result_frames += [frame]
    result += [result_frames]

    if is_save:
        if not os.path.exists(config["train"]["image_save_folder"]):
            os.mkdir(config["train"]["image_save_folder"])
        for i in range(len(result)):
            for j in range(len(result[i])):
                image_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'-{}-{}.jpg'.format(i, j)
                if iteration is not None:
                    image_name = str(iteration) + image_name

                cv2.imwrite(os.path.join(config["train"]["image_save_folder"], image_name), result[i][j])

    return result




