# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from PIL import Image

image_train_path = './mnist_data_jpg/mnist_train.jpg_60000/'
label_train_path = './mnist_data_jpg/mnist_train.jpg_60000.txt'
tfRecord_train = './data/mnist_train.tfrecords'
image_test_path = './mnist_data_jpg/mnist_test_jpg_10000/'
label_test_path = './mnist_data_jpg/mnist_test_jph_10000.txt'
tfRecord_test = '/data/mnist_test.tfrecords'
data_path = './data'
def main():
    generate_tfRecord()


def write_tfRecord(tfRecordName, image_path, label_path):
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    num_pic = 0
    f = open(label_path, 'r')
    contents = f.readlines()
    for content in contents:
        value = content.split()
        img_path =image_path + value[0]
        img = Image.open(img_path)
        img_raw = img.tobytes()
        labels = [0] * 10
        labels[int(value[1])] = 1
        example = tf.train.Example(features = tf.train.Features(feature = {
            'img_raw':tf.train.Feature(bytes_list = tf.train.BytesList(value = [img_raw])),
            'label':tf.train.Feature(int64_list = tf.train.Int64List(value = labels))
        }))
        writer.write(example.SerializeToString())
        num_pic += 1
        print("num of pic:{}".format(num_pic))
    writer.close()
    print("write tfrecord successfully")


def generate_tfRecord():
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        print("Create successfully")
    else:
        print("Exist")
    write_tfRecord(tfRecord_train, image_train_path, label_train_path)#将照片写入tfRecord_train中(以tfrecords的格式)
    write_tfRecord(tfRecord_test, image_test_path, label_test_path)


def readtfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features = {
                                           'label':tf.FixedLenFeature([10], tf.int64),
                                           'img_raw':tf.FixedLenFeature([], tf.string)
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img.set_shape([784])
    img = tf.cast(img, tf.float32) * (1./255)
    label = tf.case(features['label'], tf.float32)
    return img, label


def get_tfRecord(num, isTrain = True):
    if isTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    img,label = readtfRecord(tfRecord_path)
    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=num, num_threads=2, capacity=1000, min_after_dequeue=700)#batch_size:每次取num组，如果capacity少于700，从总样本中提取填满1000
    return img_batch, label_batch