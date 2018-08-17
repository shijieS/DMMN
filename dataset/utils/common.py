import numpy as np


def get_cx_cy_w_h(bboxes):
    bboxes = np.array(bboxes, dtype=np.float)
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    return cx, cy, w, h