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


str2bool = lambda v: v.lower() in ("yes", "true", "t", "1")

cfg = config["train"]

parser = argparse.ArgumentParser(description='Single Shot Detector and Tracker Train')
parser.add_argument('--version', default='v1', help='current version')

parser.add_argument('--basenet', default=cfg['base_net_weights'], help='resnet weights file')
parser.add_argument('--batch_size', default=cfg['batch_size'], type=int, help='Batch size for training')
parser.add_argument('--resume', default=cfg['resume'], type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=cfg['num_workers'], type=int, help='Number of workers used in dataloading')
parser.add_argument('--start_epoch', default=cfg['start_epoch'], type=int, help='end of iteration')
parser.add_argument('--end_epoch', default=cfg['end_epoch'], type=int, help='begin of iteration')
parser.add_argument('--lr_decay_per_epoch', default=cfg['lr_decay_per_epoch'], type=list, help='learning rate decay')
parser.add_argument('--cuda', default=config['cuda'], type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=cfg['learning_rate'], type=float, help='initial learning rate')
parser.add_argument('--momentum', default=cfg['momentum'], type=float, help='momentum')
parser.add_argument('--weight_decay', default=cfg['weight_decay'], type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=cfg['gamma'], type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=cfg['log_iters'], type=bool, help='Print the loss at each iteration')
parser.add_argument('--tensorboard', default=cfg['tensorboard'], type=str2bool, help='Use tensor board x for loss visualization')
parser.add_argument('--port', default=cfg['port'], type=int, help='set vidom port')
parser.add_argument('--send_images', default=cfg['send_images'], type=str2bool, help='send the generated images to tensorboard')
parser.add_argument('--log_save_folder', default=cfg['log_save_folder'], help='Location to save checkpoint models')
parser.add_argument('--weights_save_folder', default=cfg['weights_save_folder'], help='Location to save network weights')

parser.add_argument('--dataset_path', default=config['dataset_path'], help='ua dataset root folder')

args = parser.parse_args()

# load dataset
dataset = UATrainDataset(spatial_transform=Transforms())

epoch_size = len(dataset) // args.batch_size
start_iter = args.start_epoch * epoch_size
end_iter = args.end_epoch * epoch_size + 10

# re-calculate the learning rate
step_values = [i*epoch_size for i in args.lr_decay_per_epoch]

# init tensorboard
if args.tensorboard:
    from tensorboardX import SummaryWriter
    if not os.path.exists(args.log_save_folder):
        os.mkdir(args.log_save_folder)
    writer = SummaryWriter(log_dir=args.log_save_folder)

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

# creat the network
ssdt_net = SSDT.build("train")
net = ssdt_net

if args.cuda:
    net = torch.nn.DataParallel(ssdt_net, device_ids=[0])
    cudnn.benchmark = True

if args.resume:
    print("Resuming training, loading {}...".format(args.resume))
    ssdt_net.load_weights(args.resume)
else:
    print("Loading base network...")
    ssdt_net.load_base_weights(args.basenet)

if args.cuda:
    net = net.cuda()

# create optimizer
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

# create loss criterion
criterion = SSDTLoss()


# train function
def train():
    net.train()
    batch_iterator = None

    data_loader = data.DataLoader(dataset=dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  pin_memory=False)

    step_index = 0
    current_lr = args.lr
    # init learning rate for the resume
    for iteration in range(start_iter):
        if iteration in step_values:
            step_index += 1
            current_lr = adjust_learning_rate(optimizer, args.gamma, step_index)

    # start training
    batch_iterator = None
    for iteration in range(start_iter, end_iter):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
            all_epoch_loss = []

        # adjust learning rate
        if iteration in step_values:
            step_index += 1
            current_lr = adjust_learning_rate(optimizer, args.gamma, step_index)

        # reading item
        frames_1, bboxes_1, motion_parameters_1, motion_possibility_1, times_1,\
    frames_2, bboxes_2, motion_parameters_2, motion_possibility_2, times_2,\
    similarity_matrix = next(batch_iterator)

        if args.cuda:
            pass
        else:
            pass

        # forward
        t0 = time.time()
        out = net(frames_1, bboxes_1, motion_parameters_1, motion_possibility_1, times_1)

        # loss
        optimizer.zero_grad()
        loss = criterion(out)

        # backward
        loss.backward()
        optimizer.step()
        t1 = time.time()

        all_epoch_loss += [loss.data.cpu()]

        # console logs
        if iteration % 10 == 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ', ' + repr(epoch_size) + ' || epoch: %.4f ' % (iteration / (float)(epoch_size)) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')

        # tensorboard logs
        if args.tensorboard:
            pass

        # weights save
        if iteration % args.save_weights_iteration == 0:
            print('Saving weights, iter: {}', iteration)
            torch.save(ssdt_net.state_dict(),
                       os.path.join(args.weights_save_folder,
                                    'ssdt' + repr(iteration) + '.pth'))






def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()