#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

import os
from tqdm import trange
import argparse
import glob
import cv2
import pandas as pd
from draw_utils.DrawBoxes import DrawBoxes
import numpy as np


parser = argparse.ArgumentParser(description='The tools for check mot result in mot_folder')
parser.add_argument('--version', default='v1', help='version')
parser.add_argument('--mot_result_folder', default='/media/ssm/data/dataset/uadetrac/test_logs/mot-1', help='the image folder')
parser.add_argument('--image_folder', default='/media/ssm/data/dataset/uadetrac/Insight-MVT_Annotation_Train', help='the image folder')

args = parser.parse_args()


def play(mot_folder, image_folder):
    sequence_name = [os.path.basename(s) for s in glob.glob(os.path.join(image_folder, "*"))]
    for s in sequence_name:
        print("{} :================".format(s))
        s_image_folder = os.path.join(image_folder, s)
        s_mot_file = os.path.join(mot_folder, s+".txt")
        if not os.path.exists(s_mot_file):
            print("cannot find {}".format(s_mot_file))
            continue

        data = pd.read_csv(s_mot_file, header=None)
        data_group = data.groupby(by=[0])

        frame_index_list = sorted(data_group.groups.keys())
        for frame_index in frame_index_list:
            print(frame_index)
            value = data.iloc[data_group.groups[frame_index], :].values

            ids = value[:, [1]].astype(int).squeeze(1)
            bboxes = value[:, [2, 3, 4, 5]]
            conf = value[:, [6]].squeeze(1)

            image_path = os.path.join(s_image_folder, "img{0:05}.jpg".format(frame_index))
            image = cv2.imread(image_path)
            if image is None:
                print("cannot find {}".format(image_path))

            bboxes[:, [2, 3]] += bboxes[:, [0, 1]]

            if image is None:
                continue
            bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], a_min=0, a_max=image.shape[1])
            bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], a_min=0, a_max=image.shape[0])
            bboxes_valid = (bboxes[:, [2, 3]] - bboxes[:, [0, 1]] == 0).sum(axis=1) == 0
            bboxes = bboxes[bboxes_valid, :]

            colors = [DrawBoxes.get_random_color(i) for i in ids[bboxes_valid]]
            texts = ["{}".format(round(c, 2)) for c in conf]

            DrawBoxes.cv_draw_mult_boxes(image, bboxes, colors=colors, texts=texts)

            cv2.imshow("result", image)
            cv2.waitKey(-1)




if __name__ == "__main__":
    play(args.mot_result_folder, args.image_folder)
