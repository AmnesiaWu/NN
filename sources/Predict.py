import tensorflow as tf
import numpy as np
import os
from PIL import Image

img_size = 64
num_channels = 3
path = r'D:\py_pro\test\test2.jpg'
images = []

image = Image.open(path)
image = image.resize((img_size, img_size), Image.ANTIALIAS)#消除锯齿的方法处理图片
image = np.array(image)
image = image.astype(np.float32)
image = np.multiply(image, 1 / 255)
image = image.reshape(1, img_size, img_size, num_channels)#使得img为4维

sess = tf.Session()

saver = tf.train.import_meta_graph(r'../cats_dogs_model/cats_dogs.ckpt-5000.meta')# 导入模型的网络结构
saver.restore(sess, r'../cats_dogs_model/cats_dogs.ckpt-5000')
graph = tf.get_default_graph()
y_pre = graph.get_tensor_by_name("y_pre:0")#需要加:0 表示节点y_pre的第一个输出张量
x = graph.get_tensor_by_name("x:0")
res = sess.run(y_pre, feed_dict={x:image})
labels = ['dogs', 'cats']
print(labels[res.argmax()])