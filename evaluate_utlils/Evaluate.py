#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

import motmetrics as mm
import glob
import os
from motmetrics.apps.eval_motchallenge import compare_dataframes
from collections import OrderedDict
from pathlib import Path

class Evaluate:
    @staticmethod
    def evaluate_from_path(
            gt_path,
            test_path
    ):
        """
        Evaluate from the given path
        :param gt_path: the ground truth path
        :param test_path: the testing path
        :return: the evluate result formatted with Dataframe
        """
        # 1. check the existence of the input files
        if not os.path.exists(gt_path):
            raise FileNotFoundError("Cannot find files: {}".format(gt_path))

        if not os.path.exists(test_path):
            raise FileNotFoundError("Cannot find files: {}".format(test_path))

        # 2. read the data
        sequence_name = os.path.splitext(test_path)[0]
        gt = OrderedDict([(sequence_name,
                           mm.io.loadtxt(gt_path, fmt="MOT16"))])