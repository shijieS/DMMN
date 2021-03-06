#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, config):
        super(PriorBox, self).__init__()
        self.frame_size = config['frame_size']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(config['frame_work']['aspect_ratios'])
        self.variance = config['frame_work']['variance'] or [0.1]
        self.feature_maps = config['frame_work']['feature_maps']
        self.min_sizes = config['frame_work']['min_sizes']
        self.max_sizes = config['frame_work']['max_sizes']
        self.steps = config['frame_work']['steps']
        self.aspect_ratios = config['frame_work']['aspect_ratios']
        self.scales = config['frame_work']['boxes_scales']
        self.clip = config['frame_work']['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.frame_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.frame_size
                mean += [cx, cy, s_k, s_k]

                for s in self.scales[k]:
                    s_k_s = (self.min_sizes[k] + (self.max_sizes[k] - self.min_sizes[k]) * s) / self.frame_size
                    mean += [cx, cy, s_k_s, s_k_s]

                # s_k_4 = (self.min_sizes[k] + (self.max_sizes[k] - self.min_sizes[k]) / 4.0) / self.frame_size
                # s_k_3 = (self.min_sizes[k] + (self.max_sizes[k] - self.min_sizes[k]) / 1.2)/ self.frame_size
                # s_k_2 = (self.min_sizes[k] + (self.max_sizes[k] - self.min_sizes[k]) / 2.0) / self.frame_size
                # mean += [cx, cy, s_k_2, s_k_2]
                # mean += [cx, cy, s_k_3, s_k_3]
                # mean += [cx, cy, s_k_4, s_k_4]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.frame_size))
                mean += [cx, cy, s_k_prime, s_k_prime]


                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
