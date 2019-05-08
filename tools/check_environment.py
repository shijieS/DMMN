#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

from config import config
import warnings
import os

def check_weights():
    pass


def ua_check_converted_mot():
    phase = config['phase']
    dataset_name = config['dataset_name']
    if phase == "train" and dataset_name == "UA-DETRAC":
        # start to check
        ua_root = config['dataset_path']
        if not os.path.exists(os.path.join(config['dataset_path'], 'DETRAC-Train-Annotations-MOT')):
            warnings.warn("cannot find {} in the dataset directory, try to fixing ...".format('DETRAC-Train-Annotations-MOT'))
            from dataset.tools import ConvertMat2UA
            ConvertMat2UA.run(
                mat_folder=os.path.join(
                    ua_root,
                    'DETRAC-Train-Annotations-MAT'),
                save_folder=os.path.join(
                    ua_root,
                    'DETRAC-Train-Annotations-MOT'
                )
            )


if __name__ == "__main__":
   ua_check_converted_mot()
