#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

import torch
from torch.autograd import Function
from ..utils.box_utils import decode, decode_with_frames, nms, nms_with_frames
from config import config
from motion_model import MotionModel



class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh, exist_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = config["frame_work"]['variance']
        self.exist_thresh = exist_thresh

    def forward_one(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output

    def forward(self, param, p_c, p_e, priors, times):
        # param.view(param.size(0), -1, config["num_motion_model_param"]),  # parameter predicts
        # # self.softmax(p_c.view(p_c.size(0), -1, 2)),  # motion possibility
        # self.softmax(p_e.view(p_e.size(0), -1, self.num_classes)),  # classification possiblity
        # self.priors  # default boxes

        loc_datas = MotionModel.get_bbox_by_frames_pytorch(param, times)

        num = loc_datas.size(0)  # batch size
        num_priors = priors.size(0)
        num_frames = times.size(1)
        param_shape_1 = param.size(2)
        param_shape_2 = param.size(3)

        output_boxes = torch.zeros(num, self.num_classes, num_frames, self.top_k, 4)
        output_p_e = torch.zeros(num, self.num_classes, num_frames, self.top_k)
        output_params = torch.zeros(num, self.num_classes, self.top_k, param_shape_1, param_shape_2)
        output_p_c = torch.zeros(num, self.num_classes, self.top_k)
        conf_preds = p_c.squeeze(1).view(num, num_priors,
                                    self.num_classes).transpose(2, 1)
        conf_exists = p_e.transpose(3, 2)

        # Decode predictions into bboxes.
        decoded_locs = decode_with_frames(loc_datas, priors, self.variance)
        for i in range(num):
            # For each class, perform nms
            conf_scores = conf_preds[i]
            decoded_boxes = decoded_locs[i, :]
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                exists = conf_exists[i, :, cl, c_mask]

                if scores.dim() == 0:
                    continue
                boxes = decoded_boxes[:, c_mask, :]

                # if there are exists the reasonable boxes.
                print(c_mask.sum().item())
                if c_mask.sum().item() > 0:
                    # idx of highest scoring and non-overlapping boxes per class
                    ids, count = nms_with_frames(boxes, scores, p_e[i, :, c_mask, cl], self.nms_thresh, self.top_k, self.exist_thresh)

                    output_boxes[i, cl, :, :count, :] = boxes[:, ids[:count]]
                    output_p_e[i, cl, :, :count] = exists[:, ids[:count]]
                    output_params[i, cl, :count, :] = param[i, c_mask, :][ids[:count], :]
                    output_p_c[i, cl, :count] = scores[ids[:count]]
        output_p_c_1 = output_p_c.contiguous().view(num, -1)
        _, idx = output_p_c_1.sort(1, descending=True)
        _, rank = idx.sort(1)
        mask = rank > self.top_k
        output_p_c_1.masked_fill_(mask, 0)

        return (output_params, output_p_c, output_p_e, output_boxes)
        # return output_boxes
