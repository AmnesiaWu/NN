# -*- coding:utf8 -*-
import string
from captcha.image import ImageCaptcha
import random
from PIL import Image
import numpy as np
import os
import glob
from sklearn.utils import shuffle

class Dataset(object):
    def __init__(self, images, labels):
        self.num_example = images.shape[0]
        self.images = images
        self.labels = labels
        self.epoch = 0
        self.epoch_index = 0
    def next_batch(self, batch_size):
        start = self.epoch_index
        self.epoch_index += batch_size
        if self.epoch_index > self.num_example:
            start = 0
            self.epoch += 1
            self.epoch_index = batch_size
            assert batch_size <= self.num_example
        end = self.epoch_index
        return self.images[start:end], self.labels[start:end]

def gen_captcha_set(width, height, len_digits, num_train): # 获得总的训练集
    images = []
    labels = []
    characters = string.digits # 0123456789
    print('正在加载数据集......')
    path = r'D:\py_pro\test\pic\captcha'
    path = os.path.join(path, '*g')
    files = glob.glob(path)
    for file in files:
        str_capt = os.path.basename(file)
        str_std = str_capt[:5]
        image = Image.open(file)
        image = np.array(image.convert('L'))
        image = image.flatten()
        image = image.astype(np.float32)
        image = np.multiply(image, 1 / 255)
        images.append(image)
        label = np.zeros(40)
        counter = 0
        for j in range(len_digits):
            index = characters.index(str_std[j])
            label[counter + index] = 1
            counter += 10
        labels.append(label)
    print('已成功加载数据集')
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def read_train_sets(width, height, len_digits, validation, num_train): # 把训练集分为测试集和训练集
    class Datasets(object):
        pass
    data_sets = Datasets()
    vali = int(validation * num_train)
    images, labels = gen_captcha_set(width, height, len_digits, num_train)
    images, labels = shuffle(images, labels)
    images_train = images[vali:]
    labels_train = labels[vali:]

    images_test = images[:vali]
    labels_test = labels[:vali]

    data_sets.train = Dataset(images_train, labels_train)
    data_sets.test = Dataset(images_test, labels_test)
    return data_sets