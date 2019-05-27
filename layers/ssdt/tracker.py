#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

import os
import torch
from torch.autograd import Variable
from layers.ssdt import SSDT
import cv2
import numpy as np
from draw_utils.DrawBoxes import DrawBoxes
import time


class Config:
    tracker_name = "SSDT"
    tracker_version = "V1"
    cuda = True
    weigth_file = "./weights/ssdt.pth"
    frame_num = 16
    frame_height = 168
    frame_width = 168
    pixel_mean = [104, 117, 123]
    frame_scale = 2
    detect_conf_thresh = 0.5
    show_result = True
    category_map = {
        1: "bus",
        2: "car",
        3: "others",
        4: "van"
    }


class Recorder:
    pass

class Tracker:
    def __init__(self):
        #0. set torch cuda configure
        if torch.cuda.is_available():
            if Config.cuda:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            if not Config.cuda:
                print("WARNING: It looks like you have a CUDA device, but aren't " +
                      "using CUDA.\nRun with --cuda for optimal training speed.")
                torch.set_default_tensor_type('torch.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        #1. create a SSDT
        self.ssdt_net = SSDT.build("test")
        self.net = self.ssdt_net
        if Config.cuda:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.ssdt_net)

        #2. load weight
        if not os.path.exists(Config.weigth_file):
            raise FileNotFoundError("cannot find {}".format(Config.weigth_file))
        else:
            print("Loading the network")
            self.ssdt_net.load_weights(Config.weigth_file)

        self.net.eval()

        #3. create a recorder
        self.recorder = Recorder()
        self.save_frame_index = 0

    def update(self, frames, times):
        """
        frames is a list of opencv images
        :param input_frames: a list of opencv images
        :param input_times: a list of times
        :return: None
        """

        # 1. format frames and times
        if frames is None or times is None:
            return

        selected_indexes = np.arange(0, Config.frame_num) * Config.frame_scale
        input_frames = [frames[i] for i in selected_indexes]
        input_times = times

        if len(input_frames) != Config.frame_num:
            raise AssertionError("number of frames or times should be {}".format(Config.frame_num))

        input_frames = [torch.from_numpy(cv2.resize(f, (Config.frame_width, Config.frame_height)).astype(np.float32) - Config.pixel_mean)
                        for f in input_frames]

        input_times = torch.from_numpy(np.array(input_times))[None, :]
        frames_input = torch.stack(input_frames, dim=0)[None, :].permute(0, 4, 1, 2, 3)
        with torch.no_grad():
            if Config.cuda:
                frames_input = Variable(frames_input.float().cuda())
                times_input = Variable(input_times.float().cuda())
            else:
                frames_input = Variable(frames_input.float())
                times_input = Variable(input_times.float())

        # 2. get the image results
        start_time = time.time()
        output_params, output_p_c, output_p_e, output_boxes = self.net(frames_input, times_input)
        print("fps is {} sec".format(len(frames) / (time.time() - start_time)))


        # 3. update recorder
        # output_p_c_mask = output_p_c > Config.detect_conf_thresh
        output_p_e = output_p_e.permute(0, 1, 3, 2)
        output_p_c = output_p_c.permute(0, 2, 1)
        output_boxes = output_boxes.permute(0, 1, 3, 2, 4)

        for c in range(1, output_p_c.size(2)):
            mask = output_p_c[0, :, c] > Config.detect_conf_thresh
            if mask.sum() == 0:
                continue

            boxes = output_boxes[0, c, mask, :, :]
            p_c = output_p_c[0, mask, c]
            p_e = output_p_e[0, 1, mask, :]

            if Config.show_result:
                DrawBoxes.draw_ssdt_result(frames, boxes, p_c, p_e, category=Config.category_map[c])

        if Config.show_result:
            for frame in frames:
                cv2.imshow("result", frame)
                cv2.waitKey(20)
                # cv2.imwrite("result/{0:08}.png".format(self.save_frame_index), frame)
                self.save_frame_index += 1

            pass



