import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from config import config
from .models.prior_box import PriorBox
from .models.detection import Detect
import os
from .utils.generate_model import generate_resnext101
from utils import show_feature_map


class SSDT(nn.Module):

    def __init__(self, phase, base, head):
        super(SSDT, self).__init__()
        self.phase = phase
        self.num_classes = config["num_classes"]
        self.num_params = config["num_motion_model_param"]
        self.priorbox = PriorBox(config)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.input_frame_num = config["frame_max_input_num"]//2

        # base network
        self.base = base

        # localization and confidence network
        self.param_layers = nn.ModuleList(head[0])
        self.p_m_layers = nn.ModuleList(head[1])
        self.p_c_layers = nn.ModuleList(head[2])


        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(config["num_classes"], 0, 200, 0.01, 0.45)

        # init the weights and bias
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, times=None):
        sources = list()
        param = list()
        p_m = list()
        p_c = list()

        # apply resnet
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        show_feature_map(x, 'conv_1')

        x = self.base.layer1(x)
        show_feature_map(x, 'conv_2')
        sources += [x]
        x = self.base.layer2(x)
        show_feature_map(x, 'conv_3')
        sources += [x]
        x = self.base.layer3(x)
        show_feature_map(x, 'conv_4')
        sources += [x]
        x = self.base.layer4(x)
        show_feature_map(x, 'conv_5')
        sources += [x]
        x = self.base.avgpool(x)
        sources += [x]

        # apply multibox head to source layers
        i = 0
        for (x, p, m, c) in zip(sources, self.param_layers, self.p_m_layers, self.p_c_layers):
            show_feature_map(p(x), 'param{}'.format(i))
            show_feature_map(m(x), 'p_m{}'.format(i))
            show_feature_map(c(x), 'p_c{}'.format(i))
            param.append(p(x).squeeze_(dim=2).permute(0, 2, 3, 1).contiguous())
            p_m.append(m(x).squeeze_(dim=2).permute(0, 2, 3, 1).contiguous())
            p_c.append(c(x).squeeze_(dim=2).permute(0, 2, 3, 1).contiguous())

        param = torch.cat([o.view(o.size(0), -1) for o in param], 1)
        p_m = torch.cat([o.view(o.size(0), -1) for o in p_m], 1)
        p_c = torch.cat([o.view(o.size(0), -1) for o in p_c], 1)

        param = param.view(param.size(0), -1, 4, config["num_motion_model_param"] // 4)
        p_m = p_m.view(p_m.size(0), -1, self.input_frame_num, 2).permute(0, 2, 1, 3).contiguous()
        p_c = p_c.view(p_c.size(0), -1, self.input_frame_num, self.num_classes).permute(0, 2, 1, 3).contiguous()
        if self.phase == "test":
            output = self.detect(
                param,
                self.softmax(p_c),
                self.priors,
                times
            )
        else:
            output = (
                param,
                p_m,
                p_c,
                self.priors
            )
        return output

    def load_weights(self, weight_file):
        other, ext = os.path.splitext(weight_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(weight_file,
                                            map_location=lambda storage, loc: storage))
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
    def build(phase):
        if phase not in ['train', 'test']:
            print("ERROR: Phase: " + phase + " not recognized")
            return

        # build ResnNeXt 101
        base_net = generate_resnext101(
            config["num_classes"],
            frame_size=config["frame_size"],
            num_frames=config["frame_max_input_num"]//2,
            cuda=config["cuda"])

        # build parameter layers, possibility motion layers, possibility classification layers.
        param_layers = []
        p_m_layers = []
        p_c_layers = []
        num_boxes = config["frame_work"]["num_boxes"]
        num_channels_dims = config["frame_work"]["channel_dims"]
        num_temporal_dims = config["frame_work"]["temporal_dims"]
        for k, c, t in zip(num_boxes, num_channels_dims, num_temporal_dims):
            param_layers += [nn.Conv3d(in_channels=c,
                                       out_channels=k*config["num_motion_model_param"],
                                       kernel_size=(t, 3, 3),
                                       padding=(0, 1, 1),
                                       stride=(1, 1, 1),
                                       bias=True)]
            p_m_layers += [nn.Conv3d(in_channels=c,
                                     out_channels=k*(config["frame_max_input_num"]//2)*2,
                                     kernel_size=(t, 3, 3),
                                     padding=(0, 1, 1),
                                     stride=(1, 1, 1),
                                     bias=True)]
            p_c_layers += [nn.Conv3d(in_channels=c,
                                     out_channels=k * (config["frame_max_input_num"]//2) * config["num_classes"],
                                     kernel_size=(t, 3, 3),
                                     padding=(0, 1, 1),
                                     stride=(1, 1, 1),
                                     bias=True)]

        head = (param_layers, p_m_layers, p_c_layers)
        return SSDT(phase=phase,
                    base=base_net,
                    head=head)





