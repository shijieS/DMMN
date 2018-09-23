import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import config
from dataset.MotionModel import MotionModel
from .multibox_loss_param import MultiBoxLoss


class Loss(nn.Module):
    """
    This loss focus on detection.
    """

    def __init__(self):
        super(Loss, self).__init__()
        self.cuda = config["cuda"]
        # create multibox_loss from ssd
        self.multibox_loss = MultiBoxLoss(
            config["num_classes"], 0.45, True, 0,
            True, 3, 0.5, False, config["cuda"])


    def convert_to_bboxes(self, parameters, times):
        """
        current bbox's format is (cx, cy, w, h)
        """
        # N_{ba} x N_{fn} x N_{tr} x 4
        return MotionModel.get_bbox_by_frames_pytorch(parameters, times)

    def forward(self, predictions, targets, times):
        parameters, p_m_datas, p_c_datas, priors = predictions

        # convert parameters to bboxes
        loc_datas = self.convert_to_bboxes(parameters, times)

        prediction = (loc_datas, p_m_datas, p_c_datas, priors)

        # process target
        targets = [
            [
                i[t, :][i[t, :, -1] == 1].reshape(-1, 6)
                for i in targets
             ]
            for t in range(times.shape[1])
        ]

        loss_l, loss_c = self.multibox_loss(prediction, targets)

        return loss_l, loss_c





