from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
def get_w(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def get_b(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))
path = r'D:\py_pro\test\pic\captcha_test\14581.jpg'

image = Image.open(path)
image = np.array(image.convert("L"))
image = image.reshape([1, 60, 160, 1])
image = image.astype(np.float32)
w = get_w([3, 3, 1, 32])
b = get_b(32)
layer = tf.nn.conv2d(image, w, strides=[1, 1, 1, 1], padding='SAME')
layer += b
layer = tf.nn.relu(layer)
layer = tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    conv = sess.run(tf.transpose(layer, [3, 0, 1, 2]))
    print(conv.shape)

    plt.imshow(conv[0][0], cmap='gray')
    plt.show()
