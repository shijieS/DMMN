import numpy as np


class MotionModel:
    """ Motion model focuses on the change of :math:`(x_c, y_c, w, h)` with :math:`t`.
    In this model, we build a function :math:`f(t)`, for example:

    * :math:`x_c(t) = a_0 t^2 + a_1 t + a_2`
    * :math:`y_c(t) = b_0 t^2 + b_1 t + b_2`
    * :math:`w_c(t) = c_0 t^2 + c_1 t + c_2`
    * :math:`h_c(t) = d_0 t^2 + d_1 t + d_3`

    There are 12 parameters to build a model which means **parameter_num = 12**.
    """

    def __init__(self, parameter_num):
        self.parameters = np.zeros(parameter_num).reshape(4, -1)

    def fit(self, bboxes, times):
        """
        fit parameters by the frame_indexes and bboxes
        :param times: the frame index, start from 0
        :param bboxes: for each frame, the rectangle of object
        :return: the fitted parameters
        """
        pass

    def get_bbox_by_frame(self, time):
        """
        Get one bbox by the time (or frame index)
        :param time: frame_index (0-based)
        :return: one bbox
        """
        pass

    def get_bbox_by_frames(self, times):
        """
        Get bboxes by multiple times (or multiple frame indexes).
        It use :meth:`get_bbox_by_frame`

        :param times: multiple times
        :return: a list of bbox
        """

    @staticmethod
    def get_invalid_params():
        pass