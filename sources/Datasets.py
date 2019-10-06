import numpy as np
import cv2
import glob
import os
from PIL import Image
from sklearn.utils import shuffle

class DataSet(object):
    def __init__(self, images, labels, img_names, cls):
        self.num_examples = images.shape[0]
        self.images = images
        self.labels = labels
        self.img_names = img_names
        self.cls = cls
        self.epoch = 0
        self.index_in_epoch = 0
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size

        if self.index_in_epoch > self.num_examples:
            start = 0
            self.index_in_epoch = batch_size
            self.epoch += 1
            assert batch_size <= self.num_examples
        end = self.index_in_epoch
        return self.images[start : end], self.labels[start:end], self.img_names[start:end], self.cls[start:end]

def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    for fields in classes:
        index = classes.index(fields)
        path = os.path.join(train_path, fields, '*g') #路径拼接
        files = glob.glob(path) #得到文件夹下的所有文件
        for file in files:
            image = Image.open(file)
            image = image.resize((image_size, image_size), Image.ANTIALIAS)#消除锯齿的方法处理图片
            image = np.array(image)
            image = image.astype(np.float32)
            image = np.multiply(image, 1 / 255)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1
            file_name = os.path.basename(file)
            img_names.append(file_name)
            labels.append(label)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    cls = np.array(cls)
    img_names = np.array(img_names)
    return images, labels, img_names, cls

def read_train_sets(train_path, image_size, classes, validation_size):
    class DataSets(object):
        pass
    data_sets = DataSets()
    images, labels, img_names, cls = load_train(train_path, image_size, classes)
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls) #打乱顺序，但是不打乱对应顺序
    #分出训练集和测试集
    validation_size = int(validation_size * images.shape[0])
    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_img_names = img_names[:validation_size]
    validation_cls = cls[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = cls[validation_size:]

    data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
    data_sets.test = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)
    return data_sets