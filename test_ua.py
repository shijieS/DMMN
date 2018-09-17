"""
author: shijieSun
email: shijieSun@chd.edu.cn
description: This file focuses on the SSDT network testing based on the UA-DETRAC dataset
"""

import os
import torch
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from dataset.backup.ua import UATrainDataset
from config import config, cfg
from layers.ssdt import SSDT
from dataset import collate_fn
from dataset.utils import TransformsTest
from utils import show_bboxes

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
dataset = UATrainDataset(spatial_transform=TransformsTest())

# load weights
if not os.path.exists(args.resume):
    raise FileNotFoundError("cannot find {}".format(args.resume))
else:
    print("Loading the network")
    ssdt_net.load_weights(args.resume)

# test function

def test():
    net.eval()

    data_loader = data.DataLoader(dataset=dataset, batch_size=1,
                                  num_workers=1,
                                  shuffle=False,
                                  collate_fn=collate_fn,
                                  pin_memory=False)

    for _ in range(len(data_loader)):
        batch_iterator = iter(data_loader)

        frames_1, target_1, times_1, \
        frames_2, target_2, times_2, \
        similarity_matrix = \
            next(batch_iterator)

        if args.cuda:
            frames_1 = Variable(frames_1.cuda())
            with torch.no_grad():
                target_1 = [Variable(i.cuda()) for i in target_1]
                times_1 = Variable(times_1.cuda())
            frames_2 = Variable(frames_2.cuda())
            with torch.no_grad():
                target_2 = [Variable(i.cuda()) for i in target_2]
                times_2 = Variable(times_2.cuda())
        else:
            pass

        out = net(frames_1, times_1)
        batch_boxes = []
        batch_num = out.shape[0]
        for b in range(batch_num):
            boxes = []
            for i in range(config["frame_max_input_num"]//2):
                o = out[b, i, :]
                o = o.view(-1, 5)
                mask = o[:, 0] > 0.6
                l_mask = mask.unsqueeze(-1).expand_as(o)
                boxes += [o[l_mask].view(-1, 5)[:, 1:5].data]
            batch_boxes += [boxes]

        show_bboxes(frames_1, batch_boxes)
        a = 0


if __name__ == '__main__':
    test()

