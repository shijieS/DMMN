from config import config, cfg
import cv2
import torch
import numpy as np
import datetime
import os
import random
import math

def get_color_by_id(id):
    random.seed(int(id))
    return [random.randint(0, 255) for _ in range(3)]


def show_bboxes(frames, targets, is_save=True, iteration=None):
    """
    draw a corresponding rectangles to the frames
    :param frames_1: all the frames
    :param target_1: shape: [N_{ba}, 3, N_{ti}, W, H]
    :param times_1: shape: [N_{ba}, N_{ti}, (4+1+1)], where '4': bbox, '1': class_possibility, '1': motion_possibility
    :return: a list for each batch which contains all the frames
    """
    if not cfg['debug_save_image']:
        return
    N_batch, _, N_time, W, H = frames.shape
    result = []
    for n in range(N_batch):
        result_frames = []
        for t in range(N_time):
            frame = frames[n, :, t].cpu().numpy()
            frame = frame.transpose([1, 2, 0]) + config['pixel_mean']
            frame = np.clip(frame, 0, 255).astype(np.uint8).copy()

            target = targets[n][t].cpu().numpy()

            for id, bbox in enumerate(target):
                if np.any(np.isinf(bbox)):
                    continue
                if bbox[-1] != 0 and bbox[0] > -config["frame_size"]*2 and bbox[1] > -config["frame_size"]*2 and bbox[2] < config["frame_size"]*2 and bbox[3] < config["frame_size"]*2:
                    bbox[:4] *= config["frame_size"]
                    color = get_color_by_id(id)
                    frame = cv2.rectangle(frame, tuple(bbox[:2].astype(int)), tuple(bbox[2:4].astype(int)), color)
            result_frames += [frame]
    result += [result_frames]

    if is_save:
        if not os.path.exists(cfg["image_save_folder"]):
            os.mkdir(cfg["image_save_folder"])
        for i in range(len(result)):
            for j in range(len(result[i])):
                image_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'-{}-{}.jpg'.format(i, j)
                if iteration is not None:
                    image_name = str(iteration) + '-' + image_name

                cv2.imwrite(os.path.join(cfg["image_save_folder"], image_name), result[i][j])

    return result


def show_bboxes_ssdt(frames, result, is_save=True, iteration=None):
    if not is_save:
        return
    N_batch, _, N_time, W, H = frames.shape
    all_result_frames = []
    for n in range(N_batch):
        result_frames = []
        for t in range(N_time):
            frame = frames[n, :, t].cpu().numpy()
            frame = frame.transpose([1, 2, 0]) + config['pixel_mean']
            frame = np.clip(frame, 0, 255).astype(np.uint8).copy()
            bboxes = result[n][3][t, :].cpu().numpy()

            for id, bbox in enumerate(bboxes):
                if np.any(np.isinf(bbox)):
                    continue
                bbox *= config["frame_size"]
                color = get_color_by_id(id)
                frame = cv2.rectangle(frame, tuple(bbox[:2].astype(int)), tuple(bbox[2:4].astype(int)), color)
            result_frames += [frame]
    all_result_frames += [result_frames]

    if is_save:
        if not os.path.exists(cfg["image_save_folder"]):
            os.mkdir(cfg["image_save_folder"])
        for i in range(len(all_result_frames)):
            for j in range(len(all_result_frames[i])):
                image_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'-{}-{}.jpg'.format(i, j)
                if iteration is not None:
                    image_name = str(iteration) + '-' + image_name

                cv2.imwrite(os.path.join(cfg["image_save_folder"], image_name), all_result_frames[i][j])

    return all_result_frames


def convert2cv2(x):
    frame = x.cpu().data.numpy()
    min_pixel = np.min(frame)
    max_pixel = np.max(frame)

    delta = (max_pixel - min_pixel) * 255
    if delta == 0:
        delta += 1e-3
    frame = ((frame + min_pixel) / delta).astype(np.uint8)
    return frame

def show_feature_map(x, prefix="cnn"):
    if not cfg['debug_save_image']:
        return

    batch_num, feature_num, time_num, w, h = x.shape

    all_frames = [[[convert2cv2(x[n, f, t, :]) for f in range(feature_num)] for t in range(time_num)] for n in range(batch_num)]

    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    for n in range(batch_num):
        for t in range(time_num):
            prefix_ = prefix + "-b{}-t{}".format(n, t)
            cv2.imwrite(
                os.path.join(cfg["image_save_folder"],
                             time_stamp + '-' + prefix_ + ".jpg"),
                get_frame_batch(all_frames[n][t])
            )


def get_frame_batch(all_frames, gap=3):
    frame_num = len(all_frames)
    rows = math.ceil(math.sqrt(frame_num))

    frame_shape = np.array([f.shape for f in all_frames])
    max_H, max_W = np.max(frame_shape, axis=0)

    H = rows*max_H + (rows-1) * gap
    W = rows*max_W + (rows-1) * gap

    result_frame = np.zeros((H, W), dtype=np.uint8)
    start_h, start_w = 0, 0
    for i, f in enumerate(all_frames):
        h, w = f.shape
        result_frame[start_h:start_h+h, start_w:start_w+w] = f
        start_w += (gap + max_W)
        if (i+1) % rows == 0:
            start_h += (gap + max_H)
            start_w = 0
    return result_frame






