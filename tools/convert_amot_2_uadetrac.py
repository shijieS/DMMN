#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS


import os
import xml.etree.ElementTree as ET
from dataset.amot.amot_reader import AmotTestDataset
from config import config
import pandas as pd
import argparse
import cv2
import tqdm

parser = argparse.ArgumentParser(description='Single Shot Detector and Tracker Train')
parser.add_argument('--version', default='v1', help='current version')

parser.add_argument('--save_gt_folder', default="/media/ssm/data/dataset/UA_DETRAC/AMOT-EXTEND-Train-Annotations-XML-v3", help='ground truth save folder')
parser.add_argument('--save_image_folder', default="/media/ssm/data/dataset/UA_DETRAC/AMOT-EXTEND_Annotation_Train", help='image save folder')
parser.add_argument('--amot_w', default=1920, help='amot image width')
parser.add_argument('--amot_h', default=1080, help='amot image height')
parser.add_argument('--ua_w', default=960, help='ua image height')
parser.add_argument('--ua_h', default=540, help='ua image width')
parser.add_argument('--amot_root', default="/media/ssj/新加卷", help='ua dataset root')


args = parser.parse_args()

all_sequences = {
    "Town01/Clear/50/Hard_Camera_5":    "MVI_70100",
    "Town01/Clear/50/Middle_Camera_11": "MVI_70101",
    "Town01/Clear/50/Middle_Camera_3":  "MVI_70102",
    "Town02/Clear/50/Middle_Camera_11": "MVI_70200",
    "Town02/Clear/50/Easy_Camera_2":    "MVI_70201",
    "Town03/Clear/170/Middle_Camera_1": "MVI_70300",
    "Town04/Clear/170/Middle_Camera_4": "MVI_70400",
    "Town05/Clear/230/Middle_Camera_4": "MVI_70500",
}

def check():
    if not os.path.exists(args.save_gt_folder):
        os.makedirs(args.save_gt_folder)

    if not os.path.exists(args.save_image_folder):
        os.makedirs(args.save_image_folder)

def convert_label(amot_gt, sequence_name):
    wr = args.ua_w / float(args.amot_w)
    hr = args.ua_h / float(args.amot_h)

    sequence = ET.Element('sequence')
    sequence.set('name', sequence_name)
    ET.SubElement(sequence, 'sequence_attribute', {"camera_state":"unstable", "sence_weather":"sunny"})
    ET.SubElement(sequence, 'ignored_region')

    columns = [
        "frame_index", "track_id", "l", "t", "r", "b",
        "visibility", "class_id"
    ]
    converted_data = pd.read_csv(amot_gt, index_col=False)
    converted_data = converted_data.loc[:, ['frame_idx', 'id', 'l', 't', 'r', 'b', 'integrity', 'number_of_wheels']]
    converted_data.columns = columns

    # filter ignore classes
    converted_data["class_id"] = 1

    mot_data = converted_data.values

    frame_group = converted_data.groupby('frame_index').groups
    for f in sorted(frame_group.keys()):
        boxes = converted_data.iloc[frame_group[f], :].values
        frame = ET.SubElement(sequence, 'frame', {"density":str(len(boxes)), "num":str(int(f)+1)})

        target_list = ET.SubElement(frame, "target_list")
        boxes[:, [4, 5]] -= boxes[:, [2, 3]]
        boxes[:, [2, 3, 4, 5]] *= [wr, hr, wr, hr]
        for b in boxes:
            target = ET.SubElement(target_list, "target", {"id":str(int(b[1]))})
            ET.SubElement(target, "box", {
                "height":"{:.2f}".format(b[5]),
                "left":"{:.2f}".format(b[2]),
                "top":"{:.2f}".format(b[3]),
                "width":"{:.2f}".format(b[4])
            })

            ET.SubElement(target, "attribute", {
                "color":"-",
                "orientation": "-",
                "speed": "-",
                "trajectory_length": "-",
                "truncation_ratio": "{:.2f}".format(1-b[-2]),
                "vehicle_type": "Sedan"
            })

    ET.ElementTree(sequence)
    ET.ElementTree(sequence).write(os.path.join(args.save_gt_folder, "{}_v3.xml".format(sequence_name)))

def convert_image(amot_video, sequence_name):
    vc = cv2.VideoCapture(amot_video)
    save_folder = os.path.join(args.save_image_folder, sequence_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    index = 1
    while(True):
        ret, frame = vc.read()
        if not ret:
            break

        frame = cv2.resize(frame, (args.ua_w, args.ua_h))
        cv2.imwrite(os.path.join(save_folder, "img{:05}.jpg".format(index)), frame)
        index += 1
        cv2.imshow("convert", frame)
        cv2.waitKey(1)


def convert():
    for k in tqdm.tqdm(all_sequences):
        base = os.path.join(os.path.join(args.amot_root, "test"), k)
        video_path = base+".avi"
        gt_path = os.path.join(os.path.join(os.path.split(base)[0], 'gt'), os.path.split(base)[1])+".csv"
        convert_label(gt_path, all_sequences[k])
        convert_image(video_path, all_sequences[k])

if __name__ == "__main__":
    check()
    # convert_label('/media/ssj/新加卷/train/Town01/Clear/50/gt/Easy_Camera_0.csv', 'MVI_70000')
    # convert_image('/media/ssj/新加卷/train/Town01/Clear/50/Easy_Camera_0.avi', 'MVI_70000')
    convert()
