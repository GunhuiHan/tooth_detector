""" usage: partition_dataset.py [-h] [-i IMAGEDIR] [-o OUTPUTDIR] [-r RATIO] [-x]

Partition dataset of images into training and testing sets

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGEDIR, --imageDir IMAGEDIR
                        Path to the folder where the image dataset is stored. If not specified, the CWD will be used.
  -o OUTPUTDIR, --outputDir OUTPUTDIR
                        Path to the output folder where the train and test dirs should be created. Defaults to the same directory as IMAGEDIR.
  -r RATIO, --ratio RATIO
                        The ratio of the number of test images over the total number of images. The default is 0.1.
"""
import os
import re
from shutil import copyfile
import argparse
import math
import random
import json

def iterate_dir(source, dest, ratio):
    source = source.replace('\\', '/')
    dest = dest.replace('\\', '/')
    train_dir = os.path.join(dest, 'train')
    test_dir = os.path.join(dest, 'test')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    with open("ground_truth.json", "r") as gt_json:
        gt = json.load(gt_json)


    num_images = len(gt)
    num_test_images = math.ceil(ratio*num_images)

    gt_train_path = './ground_truth_train.json'
    gt_test_path = './ground_truth_test.json'

    gt_train_data = [];
    gt_test_data = [];

    for i in range(num_test_images):
        idx = random.randint(0, len(gt)-1)
        data = gt[idx]
        filename = data['filename']
        copyfile(os.path.join(source, filename + '/panorama/Originals/' + filename + '_PANO_0_1.bmp'),
                 os.path.join(test_dir, filename))
        gt_test_data.append(data) # data format :  boxes [x1, y1, x2, y2], filename
        gt.remove(data)

    for data in gt:
        filename = data['filename']
        copyfile(os.path.join(source, filename + '/panorama/Originals/' + filename + '_PANO_0_1.bmp'),
                 os.path.join(train_dir, filename))
        gt_train_data.append(data) # data format :  boxes [x1, y1, x2, y2], filename

    with open(gt_train_path, 'w') as outfile:
        json.dump(gt_train_data, outfile)
    with open(gt_test_path, 'w') as outfile:
        json.dump(gt_test_data, outfile)

def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--imageDir',
        help='Path to the folder where the image dataset is stored. If not specified, the CWD will be used.',
        type=str,
        default=os.getcwd()
    )
    parser.add_argument(
        '-o', '--outputDir',
        help='Path to the output folder where the train and test dirs should be created. '
             'Defaults to the same directory as IMAGEDIR.',
        type=str,
        default=None
    )
    parser.add_argument(
        '-r', '--ratio',
        help='The ratio of the number of test images over the total number of images. The default is 0.1.',
        default=0.1,
        type=float)
    args = parser.parse_args()

    if args.outputDir is None:
        args.outputDir = args.imageDir

    # Now we are ready to start the iteration
    iterate_dir(args.imageDir, args.outputDir, args.ratio)


if __name__ == '__main__':
    main()
