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

    def convert_to_bboxes_list(self, parameters, times):
        result = []
        for parameter, time in zip(parameters, times):
            result += [MotionModel.get_bbox_by_frames_without_batch_pytorch(parameter, time)]

        return result

    def forward(self, predictions, targets, times):
        parameters_p, p_c_p, p_e_p, priors = predictions

        # convert parameters to bboxes
        loc_datas = self.convert_to_bboxes(parameters_p, times)

        prediction = (loc_datas, p_c_p, p_e_p, priors)

        # process target
        # bbox_with_label = [
        #     [
        #         i[t, :][i[t, :, -1] == 1].reshape(-1, 6)
        #         for i in targets[0]
        #      ]
        #     for t in range(times.shape[1])
        # ]

        loc_datas_org, parameters_t, p_c_t, p_e_t = ([target[i] for target in targets] for i in range(4))
        loc_datas_t = self.convert_to_bboxes_list(parameters_t, times)

        target = (loc_datas_t, p_c_t, p_e_t)

        loss_l, loss_c, loss_m = self.multibox_loss(prediction, target)

        return loss_l, loss_c, loss_m





