""" Sample TensorFlow XML-to-TFRecord converter

usage: generate_tfrecord.py [-h] [-o OUTPUT_PATH] [-i IMAGE_DIR] [-c CSV_PATH]

optional arguments:
  -h, --help            show this help message and exit

  -i IMAGE_DIR, --image_dir IMAGE_DIR
                        Path to the folder where the input image files are stored..

  -t TYPE, --type
                        train or test
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path of output TFRecord (.record) file.
  
  -c CSV_PATH, --csv_path CSV_PATH
                        Path of output .csv file. If none provided, then no file will be written.
"""

import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf
from PIL import Image
from models.research.object_detection.utils import dataset_util
from collections import namedtuple
import pickle

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample TensorFlow json-to-TFRecord converter")

parser.add_argument("-i",
                    "--image_dir",
                    help="Path to the folder where the input image files are stored. ",
                    type=str, default=None)

parser.add_argument("-t",
                    "--type",
                    help="train or test",
                    type=str, default=None)

parser.add_argument("-o",
                    "--output_path",
                    help="Path of output TFRecord (.record) file.", type=str)

parser.add_argument("-c",
                    "--csv_path",
                    help="Path of output .csv file. If none provided, then no file will be "
                         "written.",
                    type=str, default=None)
                    

args = parser.parse_args()

def json_to_csv(path):
    path_to_json = path + 'ground_truth.json'
    json_file = open(path_to_json)
    csv_list = []
    labels=[]
    for data in json_file:
        name = 'wisdom'
        box = data['box']
        labels.append(name)
        xmin=box[0]
        ymin=box[1]
        xmax=box[2]
        ymax=box[3]
        value = (data['filename'],200,200,name,xmin,ymin,xmax,ymax)
        csv_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    csv_df = pd.DataFrame(csv_list, columns=column_name)
    print('csv_df :', csv_df)
    # labels_train=list(set(labels))
    # with open("train_labels.txt", "wb") as fp:   #Pickling
    #     pickle.dump(labels_train, fp)
    return csv_df


def class_text_to_int(row_label):
    label_map_dict = {'wisdom' : 1}
    return label_map_dict[row_label]


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, type):
    with tf.gfile.GFile(os.path.join(path + type, '/{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'bmp'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(_):
    writer = tf.python_io.TFRecordWriter(args.output_path)
    path = args.image_dir
    type = args.type
    examples = json_to_csv(path)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path, type)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecord file: {}'.format(args.output_path))
    if args.csv_path is not None:
        examples.to_csv(args.csv_path, index=None)
        print('Successfully created the CSV file: {}'.format(args.csv_path))

if __name__ == '__main__':
    tf.app.run()
    