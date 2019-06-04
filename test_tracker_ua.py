#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#
from dataset.ua.ua_reader import UATestDataset
import numpy as np
from config import config
from layers.ssdt.tracker import Tracker, Config


if __name__ == "__main__":
    dataset = UATestDataset()
    tracker = Tracker("UA", "V1", config)

    index = 0
    while index < len(dataset):
        # if index != 736:
        #     index += dataset.max_frame_num_with_scale
        #     continue
        print(index)
        frames, times = dataset[index]
        # selected_indexes = np.arange(0, dataset.max_frame_num) * dataset.frame_scale
        # image_input_list = [images[i] for i in selected_indexes]
        # times_input_list = times[selected_indexes]
        result_frames = tracker.update(frames, times, index)
        index += (dataset.max_frame_num_with_scale -  Config.share_frame_num)
