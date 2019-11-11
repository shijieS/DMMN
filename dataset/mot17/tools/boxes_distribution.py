#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#
#

from dataset.mot17.mot17 import MOT17TrainDataset
import numpy as np
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class BoxesDistribution:
    """
    Boxes distribution calculation.
    """
    def __init__(self):
        print("===== 1. Open the dataset")
        self.dataset = MOT17TrainDataset()

        print("===== 2. Read All Boxes")
        self.all_boxes = self.read_all()

        print("===== 3. convert to cx, cy, w, h format")
        self.all_boxes[:, [2, 3]] -= self.all_boxes[:, [0, 1]]
        self.all_boxes[:, [0, 1]] += self.all_boxes[:, [2, 3]] / 2.0


    def read_all(self):
        self.boxes_list = []
        for p in self.dataset.data:
            ua_data = p.ua_data
            # remove 0 box
            ua_data = ua_data[ua_data[:, :, 4] > 0, :]

            # remove confidence and visibilty
            ua_data = ua_data[:, :4]
            ua_data /= np.array([p.w, p.h, p.w, p.h])

            # get all boxes
            self.boxes_list += [ua_data]

        # stack all boxes
        all_boxes = np.concatenate(self.boxes_list, axis=0)
        return all_boxes

    @staticmethod
    def draw_3D_histogram(fig, pos, x, y, bins=10, title=None, is_3d=False):
        if is_3d:
            ax = fig.add_subplot(pos, projection='3d')
            h, xedges, yedges = np.histogram2d(x, y, bins)
            xpos, ypos = np.meshgrid(xedges[:-1]+ 0.5/bins, yedges[:-1] + 0.5/bins, indexing="ij")
            surf = ax.plot_surface(xpos, ypos, h, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
            # Add a color bar which maps values to colors.
            fig.colorbar(surf, shrink=0.5, aspect=5)
        else:
            ax = fig.add_subplot(pos)
            ax.hist2d(x, y, bins=(bins, bins), cmap=plt.cm.jet)
            # plt.colorbar()

        if title is not None:
            ax.set_title(title)

    @staticmethod
    def draw_2D_histogram(fig, pos, x, bins=10, title=None):
        ax = fig.add_subplot(pos)
        ax.hist(x, bins)
        if title is not None:
            ax.set_title(title)

    def draw_cx_cy_distribution(self, fig):
        BoxesDistribution.draw_3D_histogram(fig, 231, self.all_boxes[:, 0], self.all_boxes[:, 1], bins=100, title="cx, cy distribution")

    def draw_w_h_distribution(self, fig):
        BoxesDistribution.draw_3D_histogram(fig, 232, self.all_boxes[:, 2], self.all_boxes[:, 3], bins=100, title="w, h distribution")

    def draw_w_by_h_distribution(self, fig):
        w_by_h = self.all_boxes[:, -2] / self.all_boxes[:, -1]
        BoxesDistribution.draw_2D_histogram(fig, 233, w_by_h, bins=1000, title="w by h ratio distribution")


    def get_distribution(self):
        pass

    @staticmethod
    def get_achor_boxes():
        from layers.dmmn.models.prior_box import PriorBox
        from config import config
        pb = PriorBox(config=config)
        boxes = pb.forward().numpy()
        return boxes

    @staticmethod
    def get_ious(boxes, anchor_boxes, batch_size=2000, top_k=2, min_iou=0.3):
        from math import ceil
        from tqdm import trange
        from layers.dmmn.utils.box_utils import jaccard, point_form
        import torch
        iteration = ceil(boxes.shape[0] / batch_size)
        all_valid = []
        all_boxes = []
        for i in trange(iteration):
            start = i * batch_size
            end = (i + 1) * batch_size
            b = boxes[start:end, :]
            iou = jaccard(point_form(torch.from_numpy(b).cuda()).float(),
                          point_form(torch.from_numpy(anchor_boxes[:]).cuda()).float()).cpu().numpy()
            indexes = iou.argsort(axis=0)
            # iou.sort(axis=0)
            indexes = (indexes[-top_k:, :], range(indexes.shape[1]))
            iou = iou[indexes]
            b = b[:, None, :].repeat(iou.shape[1], axis=1)[indexes]
            all_valid += [iou]
            all_boxes += [b]
        # concatenate
        all_ious = np.concatenate(all_valid)
        all_boxes = np.concatenate(all_boxes)

        # sort
        indexes = all_ious.argsort(axis=0)
        all_ious = all_ious[indexes, range(indexes.shape[1])]
        all_boxes = all_boxes[indexes, range(indexes.shape[1])]

        # get max
        all_ious = all_ious[-1, :].reshape(-1)
        all_boxes = all_boxes[-1, :].reshape(-1, 4)

        # get the distribution of all_boxes
        mask_small_ious = all_ious < min_iou
        small_boxes = all_boxes[mask_small_ious, :]

        return all_ious, small_boxes


if __name__ == "__main__":
    bd = BoxesDistribution()
    achor_boxes = BoxesDistribution.get_achor_boxes()
    all_ious, small_boxes = BoxesDistribution.get_ious(bd.all_boxes, achor_boxes)

    fig = plt.figure()
    # draw cx, cy distribution
    bd.draw_cx_cy_distribution(fig)

    # draw w, h distribution
    bd.draw_w_h_distribution(fig)

    # draw w by h ratio distribution
    bd.draw_w_by_h_distribution(fig)

    # draw iou distributions
    BoxesDistribution.draw_2D_histogram(fig, 234, all_ious, bins=1000, title="top 1 ious distributions")
    BoxesDistribution.draw_3D_histogram(fig, 235, small_boxes[:, 2], small_boxes[:, 3], bins=100,
                                        title="min iou box w, h distribution")
    BoxesDistribution.draw_3D_histogram(fig, 236, small_boxes[:, 0], small_boxes[:, 1], bins=100,
                                        title="min iou box cx, cy distribution")
    plt.show()
