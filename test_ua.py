"""
author: shijieSun
email: shijieSun@chd.edu.cn
description: This file focuses on the SSDT network testing based on the UA-DETRAC dataset
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
from dataset.ua import UATrainDataset
from config import config
from layers.ssdt import SSDT, SSDTLoss
import time
import torchvision.utils as vutils
from dataset import collate_fn
from dataset.utils import Transforms
from utils import show_bboxes

cfg = config["test"]

parser = argparse.ArgumentParser(description='Single Shot Detector and Tracker Test')
parser.add_argument('--version', default='v1', help='current version')
parser.add_argument('--cuda', default=config['cuda'], type=bool, help='Use cuda to train model')
parser.add_argument('--resume', default=cfg['resume'], type=str, help='Resume from checkpoint')

args = parser.parse_args()

# cuda configure
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


# build network
# creat the network
ssdt_net = SSDT.build("test")
net = ssdt_net

if args.cuda:
    net = net.cuda()
    net = torch.nn.DataParallel(ssdt_net)

# load the dataset
dataset = UATrainDataset(spatial_transform=Transforms())

# load weights
if not os.path.exists(args.resume):
    raise FileNotFoundError("cannot find {}".format(args.resume))
else:
    print("Loading the network")
    ssdt_net.load_weights(args.basenet)

# test function

def test():
    net.eval()

