# -*- coding: utf-8 -*-

#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

import torch


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior

def match(threshold, truths, priors, variances, labels, exists, loc_t, conf_t, exist_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    num_frame = truths.shape[0]
    num_object = truths.shape[1]
    # num_prior = priors.shape[0]
    # all_best_truth_idx = []
    # all_best_truth_overlap = []
    # all_best_prior_idx = []
    # all_best_prior_overlap = []
    # all_overlaps = []


    # 1. select the first exist frame
    first_exists = exists.topk(1, dim=0, largest=True)[1][0]
    first_exists_truths = truths.gather(0, first_exists[:, None].expand_as(truths[0, :])[None, :])[0, :]

    # 2. get the overlaps
    overlaps = jaccard(
        first_exists_truths.float(),
        point_form(priors)
    )

    # 3. get the best priors and best truths
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_prior_idx.squeeze_(dim=1)
    best_prior_overlap.squeeze_(dim=1)
    best_truth_idx.squeeze_(dim=0)
    best_truth_overlap.squeeze_(dim=0)


    # 4. ensure best priors, and every gt matches with its prior of max overlap
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    for j in range(num_object):
        best_truth_idx[best_prior_idx[j]] = j

    # 5. make the truth label including loc_t and exists
    for frame_index in range(num_frame):
        matches = truths[frame_index][best_truth_idx]  # Shape: [num_priors,4]
        exist = exists[frame_index][best_truth_idx]
        loc = encode(matches, priors, variances)
        loc_t[idx, frame_index, :] = loc  # [num_priors,4] encoded offsets to learn
        exist_t[idx, frame_index, :] = exist

    # 6. make the truth label including conf_t
    conf = labels[best_truth_idx]
    conf[best_truth_overlap < threshold] = 0  # label as background
    conf_t[idx] = conf  # [num_priors] top class label for each prior

    """
    for frame_index in range(num_frame):
        # jaccard index
        # truths[frame_index, :] = point_form(truths[frame_index, :])
        #TODO: the overlap is too small
        overlaps = jaccard(
            truths[frame_index, :].float(),
            point_form(priors)
        )
        all_overlaps += [overlaps]
        # (Bipartite Matching)
        # [1,num_objects] best prior for each ground truth
        best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
        # [1,num_priors] best ground truth for each prior
        best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
        best_truth_idx.squeeze_(0)
        best_truth_overlap.squeeze_(0)
        best_prior_idx.squeeze_(1)
        best_prior_overlap = best_prior_overlap.squeeze_(1)
        best_prior_mask = best_prior_overlap > 0
        if len(best_prior_idx[best_prior_mask]) > 0:
            best_truth_overlap.index_fill_(0, best_prior_idx[best_prior_mask], 2)  # ensure best prior
        # ensure every gt matches with its prior of max overlap
        for j in range(num_object):
            if best_prior_mask[j]:
                best_truth_idx[best_prior_idx[j]] = j

        all_best_prior_idx += [best_prior_idx]
        all_best_prior_overlap += [best_prior_overlap]
        all_best_truth_idx += [best_truth_idx]
        all_best_truth_overlap += [best_truth_overlap]

    all_best_truth_overlap = torch.stack(all_best_truth_overlap, dim=1)
    all_best_truth_idx = torch.stack(all_best_truth_idx, dim=1)
    best_truth_overlap, best_truth_frame_idx = all_best_truth_overlap.max(dim=1, keepdim=True)
    best_truth_idx = all_best_truth_idx.gather(1, best_truth_frame_idx)
    best_truth_idx.squeeze_(1)
    best_truth_overlap.squeeze_(1)
    conf = labels[best_truth_idx]

    # make truth label
    for frame_index in range(num_frame):
        matches = truths[frame_index][best_truth_idx]          # Shape: [num_priors,4]
        exist = exists[frame_index][best_truth_idx]
        loc = encode(matches, priors, variances)
        loc_t[idx, frame_index, :] = loc    # [num_priors,4] encoded offsets to learn
        exist_t[idx, frame_index, :] = exist

    # all_best_truth_overlap = torch.stack(all_best_truth_overlap, dim=1)
    # mean_best_truth_overlap = torch.sum(all_best_truth_overlap, dim=1) / torch.sum(exist_t[idx], dim=0)
    # mean_best_truth_overlap = (all_best_truth_overlap * exist_t[idx].permute([1, 0])).sum(dim=1) / exist_t[idx].sum(dim=0)
    # mean_best_truth_overlap = ((all_best_truth_overlap>=2*threshold).float() * exist_t[idx].permute([1, 0])).sum(dim=1) / exist_t[idx].sum(dim=0)
    # mean_best_truth_overlap = ((torch.exp(all_best_truth_overlap)-1) * exist_t[idx].permute([1, 0])).sum(dim=1) / exist_t[idx].sum(dim=0)
    # mean_best_truth_overlap = (all_best_truth_overlap * exist_t[idx].permute([1, 0])).max(dim=1)[0]
    # mean_best_truth_overlap = all_best_truth_overlap.max(dim=1)[0]
    # first_exists = exist_t[idx].topk(1, dim=0, largest=True)[1][0]
    # mean_best_truth_overlap = all_best_truth_overlap.gather(dim=1, index=first_exists[:, None].expand_as(all_best_truth_overlap))[:, 0]
    # print((mean_best_truth_overlap>threshold).sum())
    select_exist = exist_t[idx].permute([1, 0]).gather(1, best_truth_frame_idx)[:, 0]
    conf[best_truth_overlap < threshold] = 0  # label as background
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    """

def encode_batch(matched, priors, variances):
    priors_batch = priors[None, None, :, :].expand_as(matched)
    # dist b/t match center and prior's center

    g_cxcy = (matched[:, :, :, :2] + matched[:, :, :, 2:]) / 2 - priors_batch[:, :, :, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors_batch[:, :, :, 2:])
    # match wh / prior wh
    g_wh = (matched[:, :, :, 2:] - matched[:, :, :, :2]) / priors_batch[:, :, :, 2:]
    g_wh = torch.where(g_wh <= 0, torch.ones_like(g_wh) * 1e-6, g_wh)

    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 3)  # [num_priors,4]

def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_with_frames(loc, priors, variances):
    p = priors.expand_as(loc)
    boxes = torch.cat((
        p[:, :, :, :2] + loc[:, :, :, :2] * variances[0] * p[:, :, :, 2:],
        p[:, :, :, 2:] * torch.exp(loc[:, :, :, 2:] * variances[1])), 3)
    boxes[:, :, :, :2] -= boxes[:, :, :, 2:] / 2
    boxes[:, :, :, 2:] += boxes[:, :, :, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)+1e-12) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


def nms_with_frames(boxes, scores, p_e, overlap=0.5, top_k=200, exist_thresh=0.3):

    num_frames = boxes.size(0)

    count = 0
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep, count

    x1 = boxes[:, :, 0]
    y1 = boxes[:, :, 1]
    x2 = boxes[:, :, 2]
    y2 = boxes[:, :, 3]

    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = (scores*((p_e > exist_thresh).sum(dim=0) > 3).float()).sort(0)  # sort in ascending order
    # v, idx = scores.float().sort(0)  # sort in ascending order
    # v, idx = (num_frames*scores+p_e.sum(dim=0).float()).sort(0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals

    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    ppe = p_e.new()
    w = boxes.new()
    h = boxes.new()

    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view

        # load bboxes of next highest vals
        torch.index_select(x1, 1, idx, out=xx1)
        torch.index_select(y1, 1, idx, out=yy1)
        torch.index_select(x2, 1, idx, out=xx2)
        torch.index_select(y2, 1, idx, out=yy2)
        torch.index_select(p_e, 1, idx, out=ppe)

        other_exist_mask = ppe > exist_thresh
        current_exist_mask = p_e[:, i] > exist_thresh
        join_exist_mask = current_exist_mask[:, None].expand_as(other_exist_mask).__and__(other_exist_mask)

        # store element-wise max with next highest score
        for frame_index in range(num_frames):
            # if current_exist_mask[frame_index]:
            #     continue
            # print(xx1.shape)
            torch.clamp(xx1[frame_index, :], min=x1[frame_index, i], out=xx1[frame_index, :])
            torch.clamp(yy1[frame_index, :], min=y1[frame_index, i], out=yy1[frame_index, :])
            torch.clamp(xx2[frame_index, :], max=x2[frame_index, i], out=xx2[frame_index, :])
            torch.clamp(yy2[frame_index, :], max=y2[frame_index, i], out=yy2[frame_index, :])

        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 1, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[:, i, None].expand_as(rem_areas)
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap

        # idx_mask = (IoU.gt(overlap).float() * ppe).sum(dim=0).gt(min_vote) == 0
        IoU[1 - join_exist_mask] = 0
        # idx_mask = (IoU.sum(dim=0) / join_exist_mask.float().sum(dim=0)) <= overlap
        idx_mask = IoU.max(dim=0)[0] <= overlap
        # idx_mask = IoU.max(dim=0)[0] <= overlap

        idx = idx[idx_mask]

    return keep, count
