import tensorflow as tf
import numpy as np
SEED = 23455
BATCH_SIZE = 8
COST = 9
PROFIT = 1

rmd = np.random.RandomState(SEED)
X = rmd.rand(32, 2)
Y_ = [[x1 + x2 + (rmd.rand() / 10 - 0.05)]for (x1, x2) in X]

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

loss = tf.reduce_sum(tf.where(tf.greater(y, y_), COST * (y - y_), PROFIT * (y_ - y)))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(30000):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 500 == 0:
            print("{}次\nw:{}".format(i, sess.run(w1)))
