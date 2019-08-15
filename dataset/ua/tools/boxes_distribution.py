#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

from dataset.ua.ua import UATrainDataset
import numpy as np
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class BoxesDistribution:
    def __init__(self):
        print("===== 1. Open the dataset")
        self.dataset = UATrainDataset()

        print("===== 2. Read All Boxes")
        self.all_boxes = self.read_all()

        print("===== 3. convert to cx, cy, w, h format")
        self.all_boxes[:, [2, 3]] -= self.all_boxes[:, [0, 1]]
        self.all_boxes[:, [0, 1]] += self.all_boxes[:, [2, 3]] / 2.0


        pass


    def read_all(self):
        self.boxes_list = []
        for p in self.dataset.data:
            ua_data = p.ua_data
            # remove 0 box
            ua_data = ua_data[ua_data[:, :, 4] > 0, :]

            # remove confidence and visibilty
            ua_data = ua_data[:, :4]

            # get all boxes
            self.boxes_list += [ua_data]

        # stack all boxes
        all_boxes = np.concatenate(self.boxes_list, axis=0)
        return all_boxes

    @staticmethod
    def draw_3D_histogram(fig, pos, x, y, bins=10, title=None):

        ax = fig.add_subplot(pos, projection='3d')
        h, xedges, yedges = np.histogram2d(x, y, bins)
        xpos, ypos = np.meshgrid(xedges[:-1]+ 0.5/bins, yedges[:-1] + 0.5/bins, indexing="ij")
        surf = ax.plot_surface(xpos, ypos, h, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        # Customize the z axis.
        # ax.set_zlim(-1.01, 1.01)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        if title is not None:
            ax.set_title(title)

    @staticmethod
    def draw_2D_histogram(fig, pos, x, bins=10, title=None):
        ax = fig.add_subplot(pos)
        ax.hist(x, bins)
        if title is not None:
            ax.set_title(title)

    def draw_cx_cy_distribution(self, fig):
        BoxesDistribution.draw_3D_histogram(fig, 221, self.all_boxes[:, 0], self.all_boxes[:, 1], bins=100, title="cx, cy distribution")

    def draw_w_h_distribution(self, fig):
        BoxesDistribution.draw_3D_histogram(fig, 222, self.all_boxes[:, 2], self.all_boxes[:, 3], bins=100, title="w, h distribution")

    def draw_w_by_h_distribution(self, fig):
        w_by_h = self.all_boxes[:, -2] / self.all_boxes[:, -1]
        BoxesDistribution.draw_2D_histogram(fig, 223, w_by_h, bins=1000, title="w by h ratio distribution")

    def get_distribution(self):
        pass

if __name__ == "__main__":
    bd = BoxesDistribution()

    fig = plt.figure()
    # draw cx, cy distribution
    bd.draw_cx_cy_distribution(fig)

    # draw w, h distribution
    bd.draw_w_h_distribution(fig)

    # draw w by h ratio distribution
    bd.draw_w_by_h_distribution(fig)

    plt.show()
