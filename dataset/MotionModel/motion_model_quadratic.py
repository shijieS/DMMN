import numpy as np
from dataset.MotionModel.motion_model import MotionModel
from dataset.utils.common import get_cx_cy_w_h
import torch


class MotionModelQuadraticPoly(MotionModel):
    """ Quadratic Polynomial Motion Model :math:`f(t) = at^2 + bt + c`

    * :math:`x_c(t) = a_0 t^2 + a_1 t + a_2`
    * :math:`y_c(t) = b_0 t^2 + b_1 t + b_2`
    * :math:`w_c(t) = c_0 t^2 + c_1 t + c_2`
    * :math:`h_c(t) = d_0 t^2 + d_1 t + d_3`
    """

    def __init__(self, parameters=None):
        super(MotionModelQuadraticPoly, self).__init__(12)
        self.parameters = parameters

    def fit(self, bboxes, times=None):
        if times is None:
            times = range(len(bboxes))

        res = get_cx_cy_w_h(bboxes)
        x = times
        deg = 2
        self.parameters = np.array(
            [list(np.polyfit(x, y, deg)) for y in res])
        return self.parameters


    def get_bbox_by_frame(self, time):
        cx_cy_w_h = np.array([p[0] * time * time + p[1] * time + p[2]] for p in self.parameters)
        cx = cx_cy_w_h[0, :]
        cy = cx_cy_w_h[1, :]
        w = cx_cy_w_h[2, :]
        h = cx_cy_w_h[3, :]
        bbox = np.stack((cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0), axis=0).transpose(1, 0)
        return bbox

    def get_bbox_by_frames(self, times):
        time_array = np.stack([np.power(times, i) for i in [2, 1, 0]], axis=1)
        cx_cy_w_h = np.sum(self.parameters * time_array[:, None, :], axis=2)
        # verify the generated box is right or not
        cx_cy_w_h[np.logical_or(cx_cy_w_h[:, 2] <= 0, cx_cy_w_h[:, 3] <= 0), :] = 0

        bbox = np.concatenate([cx_cy_w_h[:, :2] - cx_cy_w_h[:, 2:] / 2., cx_cy_w_h[:, :2] + cx_cy_w_h[:, 2:] / 2.], axis=1)
        return bbox


    @staticmethod
    def get_invalid_params():
        return np.zeros((4, 3))

    @staticmethod
    def get_num_parameter():
        return 12

    @staticmethod
    def get_parameters(bboxes, times, invalid_node_rate):
        """
        Get the parameter of boxes.
        :param bboxes: (FrameId, TrackId, 4)
        :param times: Times indexes
        :param invalid_node_rate: the threshold for cacluate the parameters
        :returns: parameters: (TrackId, ParameterData)
                  motion_possibility: (trackId, possibility)

        """
        parameters = list()
        motion_posibility = list()
        frame_num, track_num, _ = bboxes.shape
        mm = MotionModelQuadraticPoly()
        for i in range(track_num):
            bbs = bboxes[:, i, :]
            mask = np.sum(bbs, axis=1) > 0
            motion_posibility += [mask]
            if sum(mask) / len(mask) < invalid_node_rate:
                parameters += [MotionModelQuadraticPoly.get_invalid_params()]
            else:
                param = mm.fit(bbs[mask, :], times[mask])
                parameters += [param]
        return np.stack(parameters, axis=0), np.stack(motion_posibility, 1).astype(float)

    @staticmethod
    def get_str(parameters):
        p = parameters[0, :]
        return "x = {:0.2f}t^2+{:0.2f}t+{:0.2f}".format(p[0], p[1], p[2])

    @staticmethod
    def get_bbox_by_frames_pytorch(parameters, times):
        times = torch.stack([torch.pow(times, 2), torch.pow(times, 1), torch.pow(times, 0)], dim=1)
        return torch.tanh(torch.sum(parameters*times[:, None, None, :].float(), dim=3))*5
