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
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from dataset.ua.ua import UATrainDataset
from config import config, cfg
from layers.ssdt import SSDT
from dataset import collate_fn
from dataset.utils import TransformsTest
from draw_utils.Converter import TypeConverter
from draw_utils.DrawBoxes import DrawBoxes
import cv2

parser = argparse.ArgumentParser(description='Single Shot Detector and Tracker Test')
parser.add_argument('--version', default='v1', help='current version')
parser.add_argument('--cuda', default=config['cuda'], type=bool, help='Use cuda to train motion_model')
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
dataset = UATrainDataset(transform=TransformsTest())

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
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  pin_memory=False)

    batch_iterator = iter(data_loader)
    for index in range(len(data_loader)):
        frames_1, target_1, times_1 = next(batch_iterator)
        if frames_1 is None:
            continue

        if args.cuda:
            frames_1 = Variable(frames_1.cuda())
            with torch.no_grad():
                target_1 = [
                    [Variable(target[j].cuda()) for j in range(4)]
                    for target in target_1]
                times_1 = Variable(times_1.cuda())
        else:
            pass

        output_params, output_p_c, output_p_e, output_boxes = net(frames_1, times_1)
        batch_boxes = []
        batch_num = output_params.shape[0]
        class_num = output_params.size(1)
        result = []
        for b in range(batch_num):
            boxes = []

            for c in range(1, class_num):
                mask = output_p_c[b, c, :] > 0
                result += [[
                    output_params[b, c, mask, :].data,
                    output_p_c[b, c, mask].data,
                    output_p_e[b, c, :, mask].data,
                    output_boxes[b, c, :, mask, :].data,
                    c
                ]]

        # draw something on the image
        for r in result:
            all_motion_parameters = r[0]
            all_p_c = r[1]
            all_p_e = r[2]
            all_bboxes_ = r[3]
            all_c = r[4]

            # draw boxes
            for i in range(frames_1.shape[2]):
                frame = TypeConverter.image_tensor_2_cv_gpu(frames_1[0, :, i, :, :])
                all_bboxes = TypeConverter.tensor_2_numpy_gpu(all_bboxes_)
                frame = cv2.resize(frame, (1920, 960))
                h, w, c = frame.shape
                all_bboxes[:, :, [0, 2]] *= w
                all_bboxes[:, :, [1, 3]] *= h

                colors = []
                texts = []
                for c, e in zip(all_p_c, all_p_e[i, :]):
                    if e > 0.5:
                        colors += [(0, 0, 255)]
                        texts += ["{:.2}, {:.2}".format(c, e)]
                    else:
                        colors += [(255, 255, 255)]
                        texts += ["NO-"]

                DrawBoxes.cv_draw_mult_boxes_with_track(frame, all_bboxes, i, colors, texts)

                if cfg['debug_save_image']:
                    if not os.path.exists(cfg["image_save_folder"]):
                        os.makedirs(cfg["image_save_folder"])
                    cv2.imshow("result", frame)
                    cv2.waitKey(10)
                    cv2.imwrite(os.path.join(cfg["image_save_folder"], "{}-{}-{}.png".format(index, result.index(r), i)), frame)


if __name__ == '__main__':
    test()
