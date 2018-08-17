import numpy as np
from dataset.utils.motion_model import MotionModel
from dataset.utils.common import get_cx_cy_w_h


class MotionModelQuadraticPoly(MotionModel):
    """ Quadratic Polynomial Motion Model :math:`f(t) = at^2 + bt + c`

    * :math:`x_c(t) = a_0 t^2 + a_1 t + a_2`
    * :math:`y_c(t) = b_0 t^2 + b_1 t + b_2`
    * :math:`w_c(t) = c_0 t^2 + c_1 t + c_2`
    * :math:`h_c(t) = d_0 t^2 + d_1 t + d_3`
    """

    def __init__(self):
        super(MotionModelQuadraticPoly, self).__init__(12)

    def fit(self, times, bboxes):
        res = get_cx_cy_w_h(bboxes)
        x = [times[0]*times[0], times[1], times[2]]
        deg = 2
        return np.array(
            [list(np.polyfit(x, y, deg)) for y in res]
        )

    def get_bbox_by_frame(self, time):
        return [p[0]*time*time + p[1]*time + p[2]
                for p in self.parameters]

    def get_bbox_by_frames(self, times):
        return [self.get_bbox_by_frame(t) for t in times]

