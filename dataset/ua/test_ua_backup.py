from dataset.ua.ua import UATrainDataset
import cv2
from draw_utils import DrawBoxes
from motion_model import MotionModel
import numpy as np
from config import config
from dataset.utils.transforms import TransformReader

def play_ua_training_data():
    dataset = UATrainDataset(transform=TransformReader)

    print("=========dataset length=========")
    print(len(dataset))

    image_wh = np.array([config["image_width"], config["image_height"], config["image_width"], config["image_height"]])

    for item in dataset:
        (frames_1, bboxes_1, motion_parameters_1, p_e_1, times_1, p_c_1, \
        frames_2, bboxes_2, motion_parameters_2, p_e_2, times_2, p_c_2, \
        similarity_matrix) = tuple(item)

        bboxes_1 *= image_wh
        bboxes_2 *= image_wh

        frames_num_1, bboxes_num_1, _ = bboxes_1.shape
        frames_num_2, bboxes_num_2, _ = bboxes_2.shape

        motion_bboxes_1 = np.stack([MotionModel(motion_parameters_1[i, :]).get_bbox_by_frames(times_1) for i in range(bboxes_num_1)], axis=1)
        motion_bboxes_2 = np.stack([MotionModel(motion_parameters_2[i, :]).get_bbox_by_frames(times_2) for i in range(bboxes_num_2)], axis=1)

        motion_bboxes_1 *= image_wh
        motion_bboxes_2 *= image_wh

        for i, (frame1, e_1, frame2, e_2) in enumerate(zip(frames_1, p_e_1, frames_2, p_e_2)):
            # draw pre frames
            # DrawBoxes.cv_draw_mult_boxes(frame1, boxes1)
            motion_colors_1 = []
            for e in e_1:
                if e == 0:
                    motion_colors_1 += [(255, 255, 255)]
                else:
                    motion_colors_1 += [(0, 0, 0)]

            DrawBoxes.cv_draw_mult_boxes_with_track(frame1, motion_bboxes_1, i, motion_colors_1)
            DrawBoxes.cv_draw_mult_boxes_with_track(frame1, bboxes_1, i)
            cv2.imshow("frame_pre", frame1)

            # draw next frames
            motion_colors_2 = []
            for e in e_2:
                if e == 0:
                    motion_colors_2 += [(255, 255, 255)]
                else:
                    motion_colors_2 += [(0, 0, 0)]
            DrawBoxes.cv_draw_mult_boxes_with_track(frame2, motion_bboxes_2, i, motion_colors_2)
            DrawBoxes.cv_draw_mult_boxes_with_track(frame2, bboxes_2, i)
            cv2.imshow("frame_next", frame2)

            cv2.waitKey(0)





if __name__ == "__main__":
    play_ua_training_data()