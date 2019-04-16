from dataset.ua.ua import UATrainDataset
import cv2
from draw_utils import DrawBoxes
from motion_model import MotionModel
import numpy as np
from config import config
from dataset.utils.transforms import TransformReader


def play_ua_training_data():
    dataset = UATrainDataset(transform=TransformReader())

    print("=========dataset length=========")
    print(len(dataset))

    for item in dataset:
        if item is None:
            continue
        frame_indexes, track_ids, bboxes, frames, times, parameters, p_e, p_c = item
        image_wh = np.array([frames[0].shape[1], frames[0].shape[0], frames[0].shape[1], frames[0].shape[0]])

        bboxes *= image_wh
        frame_num, bboxes_num, _ = bboxes.shape
        motion_bboxes = np.stack([MotionModel(parameters[i, :]).get_bbox_by_frames(times)
                                  for i in range(bboxes_num)], axis=1)
        motion_bboxes *= image_wh

        for i, (frame, e_) in enumerate(zip(frames, p_e)):
            motion_colors = []
            for e in e_:
                if e == 0:
                    motion_colors += [(255, 255, 255)]
                else:
                    motion_colors += [(0, 0, 0)]

            DrawBoxes.cv_draw_mult_boxes_with_track(frame, motion_bboxes, i, motion_colors)
            DrawBoxes.cv_draw_mult_boxes_with_track(frame, bboxes, i)
            cv2.imshow("frame", frame)
            cv2.waitKey(0)


if __name__ == "__main__":
    play_ua_training_data()