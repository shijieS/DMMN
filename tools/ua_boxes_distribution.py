import os
import argparse
from dataset.ua.ua import UATrainDataset


parser = argparse.ArgumentParser(description='The tools for summary the distribution boxes of UA-DETRAC')
parser.add_argument('--version', default='v1', help='version')
parser.add_argument('--sequence_list', default='../dataset/sequence_list_all.txt', help='the sequence list')
parser.add_argument('--dataset_root', default='/media/ssm/data/dataset/uadetrac/', help='the dataset root')

args = parser.parse_args()


def start(dataset_root, sequence_list):
    dataset = UATrainDataset(root=dataset_root,
                             sequence_list=sequence_list)

    for item in dataset:
        if item is None:
            continue


if __name__ == "__main__":
    start(args.dataset_root, args.sequence_list)