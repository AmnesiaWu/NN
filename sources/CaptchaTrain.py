# -*- coding:utf8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import sources.CaptchaCreate

def get_w(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def get_b(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def get_convolution_layer(input_, filter_size, num_channels, output_num_filters):
    w = get_w([filter_size, filter_size, num_channels, output_num_filters])
    b = get_b(output_num_filters)
    layer = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='SAME')
    layer += b
    layer = tf.nn.relu(layer)
    layer = tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer = tf.nn.dropout(layer, keep_prob=keep_pro)
    return layer

def get_flatten_layer(layer):
    num_feature = layer.get_shape()[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_feature])
    return layer, num_feature

def get_fc_layer(input_, num_input_x, num_output, use_relu = False):
    w = get_w([num_input_x, num_output])
    b = get_b(num_output)
    layer = tf.matmul(input_, w) + b
    if use_relu:
        layer = tf.nn.relu(layer)
    layer = tf.nn.dropout(layer, keep_prob=keep_pro)
    return layer

def train(num_train):
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all() # 汇总记录点
    writer = tf.summary.FileWriter(r'D:\py_pro\test\tensorboard\summary_test', sess.graph)
    for i in range(num_train):
        x_train, y_trian = data.train.next_batch(batch_size)
        x_test, y_test = data.test.next_batch(100)
        feed_train = {x:x_train, y:y_trian, keep_pro:0.7}
        sess.run(optm, feed_dict=feed_train)
        feed_test = {x: x_test, y: y_test, keep_pro: 1.0}
        result = sess.run(merged, feed_dict=feed_test) # 运行记录点
        writer.add_summary(result, i) # 将记录点添加到模型里
        if i % 200 == 0:
            _cost = sess.run(loss, feed_dict=feed_test)
            _acc = sess.run(acc, feed_dict=feed_test)
            if _acc > 0.9:
                saver.save(sess, '../Captcha_model/captcha.ckpt', global_step=i)
            print('i:{}, cost:{}, acc:{}'.format(i, _cost, _acc))

batch_size = 32
width = 160
height = 60
len_digits = 4
validation = 0.2
num_channels = 1
num_train_examples = 60000
len_characters = 10
len_output =  int(len_characters * len_digits)
filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 64

filter_size_conv3 = 3
num_filters_conv3 = 64

filter_size_conv4 = 3
num_filters_conv4 = 64

filter_size_conv5 = 3
num_filters_conv5 = 128
fc1_size = 1024

sess = tf.Session()
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, height * width * num_channels], name='x')
    y = tf.placeholder(tf.float32, shape=[None, len_output], name = 'y')
    input_x = tf.reshape(x, [-1, height, width, num_channels], 'input_x')
with tf.name_scope('keep_pro'):
    keep_pro = tf.placeholder(tf.float32, name='keep_pro')
#卷积层
with tf.name_scope('convolution_layers'):
    convolution_layer1 = get_convolution_layer(input_x, filter_size_conv1, num_channels, num_filters_conv1)
    convolution_layer2 = get_convolution_layer(convolution_layer1, filter_size_conv2, num_filters_conv1, num_filters_conv2)
    convolution_layer3 = get_convolution_layer(convolution_layer2, filter_size_conv3, num_filters_conv2, num_filters_conv3)
    #convolution_layer4 = get_convolution_layer(convolution_layer3, filter_size_conv4, num_filters_conv3, num_filters_conv4)
    # convolution_layer5 = get_convolution_layer(convolution_layer4, filter_size_conv5, num_filters_conv4, num_filters_conv5)
# 全连接层
with tf.name_scope('fc_layers'):
    fc_input, num_features = get_flatten_layer(convolution_layer3)
    fc_layer1 = get_fc_layer(fc_input, num_features, fc1_size, True)
    output = get_fc_layer(fc_layer1, fc1_size, len_output)
with tf.name_scope('output'):
    y_pre = tf.nn.softmax(output, name='y_pre')
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pre, labels =y))
optm = tf.train.AdamOptimizer(0.0001).minimize(loss)
y_re = tf.reshape(y, [-1, len_digits, len_characters])
y_pre = tf.reshape(y_pre, [-1, len_digits, len_characters])
with tf.name_scope('accuracy'):
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_re, 2), tf.argmax(y_pre, 2)), tf.float32))
with tf.name_scope('read'):
    tf.summary.scalar('accuracy', acc)
    tf.summary.scalar('loss', loss) # 添加记录点
with tf.name_scope('save'):
    saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1)
data = sources.CaptchaCreate.read_train_sets(width, height, len_digits, validation, num_train_examples)
train(500000)