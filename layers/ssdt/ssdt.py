#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from config import config
from .models.prior_box import PriorBox
from .models.detection_param import Detect
import os
from .utils.generate_model import generate_resnext101, generate_extra_model
from draw_utils import show_feature_map
from .utils import param_init


class SSDT(nn.Module):

    def __init__(self, phase, base, head, extra):
        super(SSDT, self).__init__()
        self.phase = phase
        self.num_classes = config["num_classes"]
        self.num_params = config["num_motion_model_param"]
        self.priorbox = PriorBox(config)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
            # if config['cuda']:
            #     self.priors = Variable(self.priorbox.forward().cuda())
            # else:
            #     self.priors = Variable(self.priorbox.forward())
        self.input_frame_num = config["frame_max_input_num"]

        self.base = base

        # localization and confidence network
        self.param_layers = nn.ModuleList(head[0])
        self.p_c_layers = nn.ModuleList(head[1])
        self.p_e_layers = nn.ModuleList(head[2])

        # base net
        self.conv1 = nn.ModuleList([base.conv1, base.bn1, base.relu, base.maxpool])
        self.conv2 = base.layer1
        self.conv3 = base.layer2
        self.conv4 = base.layer3
        self.conv5 = base.layer4

        # extra net
        self.conv6 = extra.layer1
        self.conv7 = extra.layer2


        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(config["num_classes"], config["test"]["detect_bkg_label"],
                                 config["test"]["detect_top_k"], config["test"]["detect_conf_thresh"],
                                 config["test"]["detect_nms_thresh"], config["test"]["detect_exist_thresh"])

        # init the weights and bias
        self.apply(param_init)

    def forward(self, x, times=None):
        sources = list()
        param = list()
        p_c = list()
        p_e = list()

        # base net
        for conv in self.conv1:
            x = conv(x)
        show_feature_map(x, 'conv_1')
        x = self.conv2(x)
        show_feature_map(x, 'conv_2')
        sources += [x]
        x = self.conv3(x)
        show_feature_map(x, 'conv_3')
        sources += [x]
        x = self.conv4(x)
        show_feature_map(x, 'conv_4')
        sources += [x]
        x = self.conv5(x)
        show_feature_map(x, 'conv_5')
        sources += [x]

        # extra net
        x = self.conv6(x)
        show_feature_map(x, 'conv_6')
        sources += [x]
        x = self.conv7(x)
        show_feature_map(x, 'conv_7')
        sources += [x]

        # apply multibox head to source layers
        i = 0
        for (x, p, m, c) in zip(sources, self.param_layers, self.p_c_layers, self.p_e_layers):
            show_feature_map(p(x), 'param{}'.format(i))
            show_feature_map(m(x), 'p_c{}'.format(i))
            show_feature_map(c(x), 'p_e{}'.format(i))
            param.append(p(x).squeeze_(dim=2).permute(0, 2, 3, 1).contiguous())
            p_c.append(m(x).squeeze_(dim=2).permute(0, 2, 3, 1).contiguous())
            p_e.append(c(x).squeeze_(dim=2).permute(0, 2, 3, 1).contiguous())

        param = torch.cat([o.view(o.size(0), -1) for o in param], 1)
        p_c = torch.cat([o.view(o.size(0), -1) for o in p_c], 1)
        p_e = torch.cat([o.view(o.size(0), -1) for o in p_e], 1)

        param = param.view(param.size(0), -1, 4, config["num_motion_model_param"] // 4)
        p_c = p_c.view(p_c.size(0), -1, 1, self.num_classes).permute(0, 2, 1, 3).contiguous()
        p_e = p_e.view(p_e.size(0), -1, self.input_frame_num, 2).permute(0, 2, 1, 3).contiguous()
        if self.phase == "test":
            output = self.detect(
                param,
                self.softmax(p_c),
                self.softmax(p_e),
                self.priors,
                times
            )
        else:
            output = (
                param,
                p_c,
                p_e
            )
        return output

    def load_weights(self, weight_file):
        other, ext = os.path.splitext(weight_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            model_data = torch.load(weight_file)
            self.load_state_dict(model_data['state_dict'])
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def load_base_weights(self, base_file):
        if os.path.splitext(base_file)[1] in ['.pkl', '.pth']:
            print('Loading base net weights into state dict...')
            model_data = torch.load(base_file)
            assert config["base_net"]["arch"] == model_data['arch']
            self.base.load_state_dict(model_data['state_dict'], strict=False)
            print('Finish')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def load_weights(self, resume):
        model_data = torch.load(resume)
        self.load_state_dict(model_data)

    @staticmethod
    def build_extra_net1(in_channel):
        torch.nn.Conv3d(in_channel, 32, kernel_size=1, stride=1, bias=False)
        torch.nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        torch.nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        torch.nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        torch.nn.Conv3d(32, in_channel, kernel_size=1, stride=1, bias=False)
        torch.nn.BatchNorm3d(in_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    @staticmethod
    def build(phase):
        if phase not in ['train', 'test']:
            print("ERROR: Phase: " + phase + " not recognized")
            return

        # build ResnNeXt 101
        base_net = generate_resnext101(
            config["num_classes"],
            frame_size=config["frame_size"],
            num_frames=config["frame_max_input_num"],
            cuda=config["cuda"])

        # build extra net
        extra_net_inplanes = base_net.layer4[2].conv3.out_channels
        extra_net = generate_extra_model(cuda=config["cuda"], inplanes = extra_net_inplanes)
        # extra_net = SSDT.build_extra_net(base_net.layer4[2].conv3.out_channels)

        # build parameter layers, possibility motion layers, possibility classification layers.
        param_layers = []
        p_c_layers = []
        p_e_layers = []

        scales = config["frame_work"]["boxes_scales"]
        aspect_ratios = config["frame_work"]["aspect_ratios"]
        num_boxes = [2 + len(_s) + 2*len(_a) for _s, _a in zip(scales, aspect_ratios)]

        # num_boxes = config["frame_work"]["num_boxes"]
        num_channels_dims = config["frame_work"]["channel_dims"]
        num_temporal_dims = config["frame_work"]["temporal_dims"]
        for k, c, t in zip(num_boxes, num_channels_dims, num_temporal_dims):
            param_layer = nn.Conv3d(in_channels=c,
                                     out_channels=k*config["num_motion_model_param"],
                                     kernel_size=(t, 3, 3),
                                     padding=(0, 1, 1),
                                     stride=(1, 1, 1),
                                     bias=True)
            p_c_layer = nn.Conv3d( in_channels=c,
                                    out_channels=k*config["num_classes"],
                                    kernel_size=(t, 3, 3),
                                    padding=(0, 1, 1),
                                    stride=(1, 1, 1),
                                    bias=True)
            p_e_layer = nn.Conv3d( in_channels=c,
                                    out_channels=k * config["frame_max_input_num"] * 2,
                                    kernel_size=(t, 3, 3),
                                    padding=(0, 1, 1),
                                    stride=(1, 1, 1),
                                    bias=True)
            if config["cuda"]:
                param_layer = param_layer.cuda()
                p_c_layer = p_c_layer.cuda()
                p_e_layer = p_e_layer.cuda()

            param_layers += [param_layer]
            p_c_layers += [p_c_layer]
            p_e_layers += [p_e_layer]


        head = (param_layers, p_c_layers, p_e_layers)
        return SSDT(phase=phase,
                    base=base_net,
                    extra=extra_net,
                    head=head)





