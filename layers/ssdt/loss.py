import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import config
from dataset.MotionModel import MotionModel
from .multibox_loss import MultiBoxLoss

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cuda = config["cuda"]
        # create multibox_loss from ssd
        self.multibox_loss = MultiBoxLoss(
            config["num_classes"], 0.2, True, 0,
            True, 3, 0.5, False, config["cuda"])

    def generate_targets_by_time(self, motion_parameters,
                                 motion_possiblity, class_possibility, time):
        pass

    def get_multibox_loss(self, predictions, targets):
        pass

    def convert_to_bboxes(self, parameters, times):
        """
        current bbox's format is (cx, cy, w, h)
        """
        n = parameters.shape[0]
        bboxes = []
        for i in range(n):
            bboxes += [
                MotionModel.get_bbox_by_frames_pytorch(
                    parameters[i, :], times[i, :])
            ]

        return torch.stack(bboxes, 0)   # N_{ba} x N_{fn} x N_{tr} x 4

    def forward(self, predictions, targets, times):
        parameters, p_m_datas, p_c_datas, priors = predictions

        # convert parameters to bboxes
        loc_datas = self.convert_to_bboxes(parameters, times)

        # split all the data by frames
        all_loss_l = []
        all_loss_c = []
        for t in range(times.shape[1]):
            loc_data = loc_datas[:, t, :]
            p_m_data = p_m_datas[:, t, :]
            p_c_data = p_c_datas[:, t, :]
            prediction = (loc_data, p_c_data, p_m_data, priors.clone())
            target = [
                i[t, :][
                    i[t, :, -1] == 1
                ].reshape(-1, 6)
                for i in targets]
            loss_l, loss_c = self.multibox_loss(prediction, target)
            if loss_l is not None:
                all_loss_l += [loss_l]
                all_loss_c += [loss_c]

        return all_loss_l, all_loss_c




