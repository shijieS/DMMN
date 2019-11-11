#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

import os
from tqdm import trange, tqdm
import argparse
import glob
import cv2


parser = argparse.ArgumentParser(description='The tools for convert images to video')
parser.add_argument('--version', default='v1', help='version')
parser.add_argument('--image_folder', default='/media/ssm/data/dataset/UA_DETRAC/test_logs/dmmn-log-0726-all-1081730/images0.4', help='the image folder')
parser.add_argument('--video_file', default='/media/ssm/data/dataset/amotd/test_logs/0808-67650/images/testTown02Clear50Easy_Camera_0.avi', help='the video file to be saved')
parser.add_argument('--video_fps', default=25, help="Video fps")
parser.add_argument('--video_height', default=1080, help="Video height")
parser.add_argument('--video_width', default=1920, help="Video width")

# parser.add_argument('--image_format', default='{}-{}-{}.png', help='image format')

args = parser.parse_args()

def convert(image_folder, video_file, fps, width, height):
    # 1. read all images
    images = sorted(glob.glob(os.path.join(image_folder, '*.jpg')), key=os.path.getmtime)
    # 2. start convert
    vw = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

    for i in trange(len(images)):
        try:
            I = cv2.imread(images[i])
            I = cv2.resize(I, (width, height))
            vw.write(I)
            cv2.waitKey(int(1000/fps))
        except:
            continue



def convert_amot_images_2_videos(image_folder):
    folders = glob.glob(os.path.join(image_folder, "*"))
    for f in tqdm(folders):
        if not os.path.isdir(f):
            continue
        video_name = f + ".avi"
        convert(f, video_name, args.video_fps, args.video_width, args.video_height)

def convert_ua_images_2_videos(image_folder):
    folders = glob.glob(os.path.join(image_folder, "*"))
    for f in tqdm(folders):
        if not os.path.isdir(f):
            continue
        video_name = f + ".avi"
        convert(f, video_name, args.video_fps, 960, 540)

if __name__ == "__main__":
    convert(args.image_folder, args.video_file, args.video_fps, args.video_width, args.video_height)
    # convert_amot_images_2_videos(args.image_folder)
    # convert_ua_images_2_videos(args.image_folder)