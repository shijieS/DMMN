#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

import argparse
import os
import numpy as np
import glob
import pandas as pd
from tqdm import trange

print('''
Convert MOT result to ua_result
Usage: convert_mat_2_ua --ua="ua root path"
''')


parser = argparse.ArgumentParser(description='UA Result Covnerter')
parser.add_argument('--mot_folder', default=r"/media/ssm/data/dataset/UA_DETRAC/test_logs/ssdt-log-0726-all-1081730/logs0.4/mot",
                    help='''mot result folder, with the following directory structure:
                    folder
                    |
                    |-- 0.1
                    |-- 0.2
                    |-- ...
                    ''')
parser.add_argument('--ua_folder', default=r"/media/ssm/data/dataset/UA_DETRAC/test_logs/ssdt-log-0726-all-1081730/logs0.4/mot-ua", help='ua result folder. This tool would create this folder with same sturcture')
parser.add_argument('--dataset_folder', default=r"/media/ssm/data/dataset/UA_DETRAC", help='ua result folder. This tool would create this folder with same sturcture')
parser.add_argument('--min_visibility', default=0.4)

args = parser.parse_args()


class ConvertTools:
    min_visibility = 0.4

    @staticmethod
    def get_max_frames(name):
        train_test_list = ["Insight-MVT_Annotation_Test/", "Insight-MVT_Annotation_Train/"]
        for k in train_test_list:
            image_folder = os.path.join(args.dataset_folder, k+name)
            if os.path.exists(image_folder):
                image_list = glob.glob(os.path.join(image_folder, "*.jpg"))
                image_indexes = [int(os.path.basename(i)[3:8]) for i in image_list]
                return max(image_indexes)
        return -1


    @staticmethod
    def generate_ua_result(mot_file, ua_folder):
        name = os.path.splitext(os.path.basename(mot_file))[0]
        data = pd.read_csv(mot_file, delimiter=',', header=None)

        row_num = ConvertTools.get_max_frames(name)
        col_num = int(data[1].max())

        data = data[data[8] > ConvertTools.min_visibility]
        data = data.values


        # create ua_data
        ua_data = np.zeros((row_num, col_num, 4))
        for d in data:
            ua_data[int(d[0]-1), int(d[1]-1), :] = d[2:6]

        # save ua_data
        ua_files = [os.path.join(ua_folder, name+"_"+k+".txt") for k in ["LX", "LY", "W", "H"]]
        for i, f in enumerate(ua_files):
            np.savetxt(f, ua_data[:, :, i], fmt="%.2f", delimiter=',')

        np.savetxt(os.path.join(ua_folder, name+"_Speed.txt"), np.array([40]), fmt="%.2f")


    @staticmethod
    def init(mot_folder, ua_folder, min_visibility):
        ConvertTools.min_visibility = min_visibility

        if not os.path.exists(mot_folder):
            raise FileNotFoundError('cannot find {}'.format(mot_folder))

        if not os.path.exists(ua_folder):
            os.mkdir(ua_folder)

        #  get the list of mot result files
        mot_files = glob.glob(os.path.join(mot_folder, '*.txt'))

        # generate the corresponding ua-detrac formated files
        for _, f in zip(trange(len(mot_files)), mot_files):
            ConvertTools.generate_ua_result(f, ua_folder)


def run_converter():
    for i in range(11):
        mot_folder = '/media/ssm/data/dataset/UA_DETRAC/test_logs/ssdt-log-0911-ua-amot-408394-2/logs{:.1f}/mot'.format(i*0.1)
        ua_folder = '/media/ssm/data/dataset/UA_DETRAC/test_logs/ssdt-log-0911-ua-amot-408394-2/ua/SSDT/{:.1f}'.format(i*0.1)
        mini_visibility = 0.3
        if not os.path.exists(ua_folder):
            os.makedirs(ua_folder)
        ConvertTools.init(mot_folder, ua_folder, mini_visibility)



if __name__ == '__main__':
    # condition
    # ConvertTools.init(args.mot_folder, args.ua_folder, args.min_visibility)
    run_converter()