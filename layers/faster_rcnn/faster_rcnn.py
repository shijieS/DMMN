#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

from torch import nn


class FasterRCNN(nn.Module):
    def __init__(self,
                 extractor,
                 rpn,
                 head,
                 loc_normalize_mean = (0., 0., 0., 0.),
                 loc_normalize_std = (0.1, 0.1, 0.1, 0.1)
                 ):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

        self.use_preset('evaluate')


    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thres = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thres = 0.05
        else:
            raise ValueError('preset must be \'visualize\' or \'evaluate\' ')

    @property
    def n_class(self):
        return self.head.n_class

    def forward(self, x, scale=1.):
        img_size = x.shape[2:]
        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
        self.rpn(h, img_size, scale)

        roi_cls_locs, roi_scores = self.head(
            h, rois, roi_indices
        )

        return roi_cls_locs, roi_scores, rois, roi_indices