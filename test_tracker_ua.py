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
import os


def run_tracker(config):
    dataset = UATestDataset()
    tracker = Tracker("UA", "V1", config)

    index = 0
    sequence_name = None
    while index < len(dataset):
        # 1. if switch video, then save and clear all tracks
        current_sequence_name = dataset.sequence_list[dataset.get_groupd_index(index)]
        if sequence_name is None:
            sequence_name = current_sequence_name
            Config.set_image_folder(
                os.path.join(config['test']['image_save_folder'], current_sequence_name)
            )

        if sequence_name != current_sequence_name:
            save_mot_folder = os.path.join(config["test"]["log_save_folder"], "mot")
            if not os.path.exists(save_mot_folder):
                os.makedirs(save_mot_folder)
            mot_file = os.path.join(save_mot_folder,
                                    "{}.txt".format(sequence_name))
            tracker.save_mot_result(mot_file, True)
            Config.set_image_folder(
                os.path.join(config['test']['image_save_folder'], current_sequence_name)
            )

        sequence_name = current_sequence_name

        # 2. get items
        frames, times, start_frame_index = dataset[index]

        # 3. update trackers
        result_frames = tracker.update(frames, times, start_frame_index)

        # 4. save mot results
        save_mot_folder = os.path.join(config["test"]["log_save_folder"], "mot")
        if not os.path.exists(save_mot_folder):
            os.makedirs(save_mot_folder)
        mot_file = os.path.join(save_mot_folder,
                                "{}.txt".format(sequence_name))
        tracker.save_mot_result(mot_file)

        index += (dataset.max_frame_num_with_scale - Config.share_frame_num)

def run_tracker_for_ua_result():
    config_list = [
        [("test", "./dataset/ua/sequence_list_test.txt", 0.4 + 0.03 * i),
         ("train", "./dataset/ua/sequence_list_train.txt", 0.4 + 0.03 * i)] for i in range(1, 11)
    ]
    log_save_folder = config["test"]["log_save_folder"]
    image_save_folder = config["test"]["image_save_folder"]
    weights_save_folder = config["test"]["weights_save_folder"]
    for item in config_list:
        config["test"]["dataset_type"] = item[0][0]
        config["test"]["sequence_list"] = item[0][1]
        config["test"]["detect_conf_thresh"] = item[0][2]
        config["test"]["log_save_folder"] = log_save_folder + str(item[0][2])
        config["test"]["image_save_folder"] = image_save_folder + str(item[0][2])
        config["test"]["weights_save_folder"] = weights_save_folder + str(item[0][2])

        if not os.path.exists(config["test"]["log_save_folder"]):
            os.makedirs(config["test"]["log_save_folder"])
        if not os.path.exists(config["test"]["image_save_folder"]):
            os.makedirs(config["test"]["image_save_folder"])
        if not os.path.exists(config["test"]["weights_save_folder"]):
            os.makedirs(config["test"]["weights_save_folder"])

        run_tracker(config)

if __name__ == "__main__":
    # run_tracker(config)
    # test data set
    run_tracker_for_ua_result()

