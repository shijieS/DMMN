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
import pprint

print('''
Combine each sequence result
Usage: ua_combine_result --result_folder="ua root path"
''')


parser = argparse.ArgumentParser(description='The fodler of result')
parser.add_argument('--result_folder', default=r"/media/ssm/data/dataset/UA_DETRAC/results/SSDT_265330")
parser.add_argument('--save_file', default=r"/media/ssm/data/dataset/UA_DETRAC/results/SSDT_265330.csv")


args = parser.parse_args()

metric_name = ['SequenceName', 'Rcll', 'Prcn', 'FAR', 'UName', 'MT', 'PT', 'ML', 'FP', 'FN', 'IDs', 'FM', 'MOTA', 'MOTP', 'MOTAL']

def combine_results(result_folder):
    sequence_list = glob.glob(os.path.join(result_folder, "*_mot_result.txt"))
    name_list = []
    all_result = []
    for s in sequence_list:
        n = os.path.basename(s)[:9]
        d = pd.read_csv(s, sep=',', header=None)
        all_result.append(d)
        name_list += [n]

        continue
    result = pd.concat(all_result).iloc[:, 1:]
    result.insert(0, 'name', name_list)
    result.columns = metric_name

    return result

if __name__ == "__main__":
    result = combine_results(args.result_folder)
    result.to_csv(args.save_file)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(result)


