#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

import numpy as np


def get_cx_cy_w_h(bboxes):
    bboxes = np.array(bboxes, dtype=np.float)
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    return cx, cy, w, h


def get_bx_by_w_h(bboxes):
    bboxes = np.array(bboxes, dtype=np.float)
    bx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    by = bboxes[:, 3]
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    return bx, by, w, h