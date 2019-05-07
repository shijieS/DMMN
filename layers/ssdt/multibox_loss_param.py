# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .utils.box_utils import match, log_sum_exp, encode_batch
from config import config


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = config["frame_work"]['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_datas_p, p_c_p, p_e_p, priors = predictions
        loc_datas_t, p_c_t, p_e_t = targets

        num = loc_datas_p.size(0)
        num_frames = loc_datas_p.shape[1]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        loc_ts = torch.Tensor(num, num_frames, num_priors, 4)
        p_c_ts = torch.LongTensor(num, 1, num_priors)
        p_e_ts = torch.Tensor(num, num_frames, num_priors)

        if self.use_gpu:
            loc_ts = loc_ts.cuda()
            p_c_ts = p_c_ts.cuda()
            p_e_ts = p_e_ts.cuda()

        for idx in range(num):
            truths = loc_datas_t[idx].float()
            labels = p_c_t[idx]
            exists = p_e_t[idx]
            defaults = priors.data

            match(self.threshold, truths, defaults, self.variance, labels, exists, loc_ts, p_c_ts, p_e_ts, idx)

        with torch.no_grad():
            loc_ts = Variable(loc_ts)
            p_c_ts = Variable(p_c_ts)
            p_e_ts = Variable(p_e_ts)

        # N_b x 1 x N_p
        pos = p_c_ts > 0

        # Localization Loss (Smooth L1)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_datas_p)   # N_b x N_f x N_p x 4
        exist_idx = p_e_ts.unsqueeze(p_e_ts.dim()).expand_as(loc_datas_p) > 0 # N_b x N_f x N_p x 4
        pos_idx = (pos_idx * exist_idx) > 0 # N_b x N_f x N_p x 4
        # encode loc_data_p
        # loc_datas_p = encode_batch(loc_datas_p, priors, self.variance)
        loc_p = loc_datas_p[pos_idx].view(-1, 4)
        loc_ts = loc_ts[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_ts, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = p_c_p.contiguous().view(-1, self.num_classes) # (N_b x N_p) x N_c
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, p_c_ts.view(-1, 1)) # N_b x 1 x N_p

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1).unsqueeze(1) # N_b x 1 x N_p
        loss_c[pos] = 0  # filter out pos boxes
        _, loss_idx = loss_c.sort(2, descending=True)
        _, idx_rank = loss_idx.sort(2)
        num_pos = pos.long().sum(2, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.shape[2] - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank) # N_b x 1 x N_p

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(3).expand_as(p_c_p) # N_b x 1 x N_p x 2
        neg_idx = neg.unsqueeze(3).expand_as(p_c_p) # N_b x 1 x N_p x 2
        conf_p = p_c_p[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = p_c_ts[(pos + neg).gt(0)]
        if len(targets_weighted) == 0 or len(conf_p) == 0:
            loss_c = conf_p.sum()
        else:
            loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # calcualte the exists loss
        pos_idx = pos.unsqueeze(3).expand_as(p_e_p)
        neg_idx = neg.unsqueeze(3).expand_as(p_e_p)
        exist_p = p_e_p[(pos_idx + neg_idx).gt(0)].view(-1, 2)
        pos_idx = pos.expand_as(p_e_ts)
        neg_idx = neg.expand_as(p_e_ts)
        targets_weighted = p_e_ts[(pos_idx + neg_idx).gt(0)].long()
        if len(targets_weighted) == 0 or len(exist_p) == 0:
            loss_e = exist_p.sum()
        else:
            loss_e = F.cross_entropy(exist_p, targets_weighted, reduction='sum')

        N = num_pos.data.sum().float()
        return loss_l / N, loss_c / N * num_frames, loss_e / N

