import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num = 1000
data_set = []
for i in range(num):
    x = np.random.normal(0.0, 0.55) # 生成高斯分布的概率密度随机数
    y = x * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    data_set.append([x, y])

x_data = [v[0] for v in data_set]
y_data = [v[1] for v in data_set]

W = tf.Variable(tf.random.uniform([1], -1, 1, name='W'))
b = tf.Variable(tf.zeros([1]), name='b')
y = W * x_data + b
loss = tf.reduce_mean(tf.square(y - y_data), name='loss') # 方差
optimizer = tf.train.GradientDescentOptimizer(0.5) # 优化器
train = optimizer.minimize(loss, name='train') # 优化：尽量使得方差最小
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('w ={}, b = {}, loss = {}'.format(sess.run(W), sess.run(b), sess.run(loss)))
    for i in range(20):
        sess.run(train)
        print('w ={}, b = {}, loss = {}'.format(sess.run(W), sess.run(b), sess.run(loss)))