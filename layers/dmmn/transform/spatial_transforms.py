#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

import numbers
import collections
import numpy as np
import torch
from PIL import Image, ImageOps
import torchvision
try:
    import accimage
except ImportError:
    accimage = None


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, item):
        for t in self.transforms:
            item = t(item)
        return item


class ToTensor(object):
    def __call__(self, item):
        # transpose to RGB
        item[0] = [torch.from_numpy(i.transpose([2, 1, 0])).float() for i in item[0]]
        item[5] = [torch.from_numpy(i.transpose([2, 1, 0])).float() for i in item[5]]

        # all to tensor
        out = []
        for i in item:
            if i == 0 or i == 5:
                continue
            if i.dtype == int:
                out += [torch.from_numpy(i).int()]
            else:
                out += [torch.from_numpy(i).float()]

        return out


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, item):
        item[0] = [torchvision.transforms.Normalize(self.mean, self.std) for i in item[0]]
        item[5] = [torchvision.transforms.Normalize(self.mean, self.std) for i in item[5]]

        return item

