

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnext import ResNeXtBottleneck
from functools import partial
from .resnext import downsample_basic_block
from ..utils import param_init

class ExtraNet(nn.Module):
    def __init__(self, block, layers, shortcut_type='B', cardinality=32, inplanes=64):
        self.inplanes = inplanes
        super(ExtraNet, self).__init__()
        self.layer1 = self._make_layer(block, 1024, layers[0], shortcut_type, cardinality, stride=2)
        self.layer2 = self._make_layer(block, 1024, layers[1], shortcut_type, cardinality, stride=2)

        self.apply(param_init)

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
