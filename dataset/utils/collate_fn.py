#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

import torch


def generate_targets(org_bboxes, motion_parameters, p_e, times, p_c):
    """
    generate a benchmark target
    :param motion_parameters: shape:[N_t
    :param p_e:
    :param times:
    :param p_c:
    :return:
    """
    track_num, _, _ = motion_parameters.shape

    # stack the possibility into the bboxes
    target = [
        org_bboxes,
        motion_parameters,
        p_c,
        p_e
    ]

    return target


def collate_fn(batch):
    frames = []               # 0
    target = []               # 1 (N_{fn} x N_{re} x (4+1+1))
    times = []                # 2

    if sum([s is not None for s in batch]) == 0:
        return None, None, None
    # split batch
    for items in batch:
        if items is None:
            continue
        # convert to tensor
        frames.append(items[3])
        target.append(generate_targets(items[2], items[5], items[6], items[4], items[7]))
        times.append(items[4].float())

    # stack batch
    return torch.stack(frames, 0).permute(0, 4, 1, 2, 3), target, torch.stack(times, 0)
