import sources.Datasets as datasets
import tensorflow as tf
import random
import numpy as np

from numpy.random import seed
from tensorflow import set_random_seed

def create_w(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_b(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def create_convolutional_layer(inputs, filter_size, num_channelss, num_filters):
    weights = create_w([filter_size, filter_size, num_channelss, num_filters])
    biases = create_b(num_filters)

    layer = tf.nn.conv2d(input=inputs, filter=weights, strides=[1, 1, 1, 1], padding="SAME") #卷积开始
    layer += biases

    layer = tf.nn.relu(layer) #激活函数

    layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME") #池化
    return layer

def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    return layer, num_features

def create_fc_layer(inputs, num_inputs, num_outputs, use_relu = False):
    weights = create_w([num_inputs, num_outputs])
    biases = create_b(num_outputs)
    layer = tf.matmul(inputs, weights) + biases
    layer = tf.nn.dropout(layer, keep_prob=0.7)
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

def train(num_iteration):
    for i in range(num_iteration):
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_test_batch, y_test_batch, _, cls_test_batch = data.test.next_batch(batch_size)
        feed_train = {x:x_batch, y_true:y_true_batch}
        feed_test = {x:x_test_batch, y_true:y_test_batch}
        sess.run(optm, feed_dict=feed_train)
        if i % 200 == 0:
            cost_ = sess.run(cost, feed_dict=feed_test)
            acc_ = sess.run(acc, feed_dict=feed_test)
            saver.save(sess, '../cats_dogs_model/cats_dogs.ckpt', global_step=i)
            print("{}times， cost:{}, accuracy:{}".format(i, cost_, acc_))
batch_size = 32
classes = ['dogs', 'cats']
num_classes = len(classes)

validation_size = 0.2 #训练集中，有20%作为测试集
img_size = 64
num_channels = 3
train_path = '../pic'

data = datasets.read_train_sets(train_path, image_size=img_size, classes= classes, validation_size=validation_size)
sess = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true,1)

filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 1024
#卷积层
layer_conv1 = create_convolutional_layer(x, filter_size_conv1, num_channels, num_filters_conv1)
layer_conv2 = create_convolutional_layer(layer_conv1, filter_size_conv2, num_filters_conv1, num_filters_conv2)
layer_conv3 = create_convolutional_layer(layer_conv2, filter_size_conv3, num_filters_conv2, num_filters_conv3)
#全连接层
layer_flat, num_featrues = create_flatten_layer(layer_conv3)
layer_fc1 = create_fc_layer(inputs=layer_flat, num_inputs=num_featrues, num_outputs=fc_layer_size, use_relu=True)
layer_fc2 = create_fc_layer(inputs=layer_fc1, num_inputs=fc_layer_size, num_outputs=num_classes)

y_pre = tf.nn.softmax(layer_fc2, name='y_pre')
y_pre_cls = tf.argmax(y_pre, 1)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pre, labels=y_true))
optm = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
acc = tf.reduce_mean(tf.cast(tf.equal(y_true_cls, y_pre_cls), tf.float32))
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=10,keep_checkpoint_every_n_hours=1)
train(50000)