import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
BATCH_SIZE = 30
seed = 2
rdm = np.random.RandomState(seed)
X = rdm.randn(300, 2)
Y_ = [int(x0 * x0 + x1 * x1 < 2) for (x0, x1) in X]
Y_c = ['red' if y else 'blue' for y in Y_]

X = np.vstack(X).reshape(-1, 2)
Y_ = np.vstack(Y_).reshape(-1, 1)
print(X)
print(Y_)
print(Y_c)

plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
plt.show()


def get_w(shape, regular):
    w = tf.Variable(tf.random.normal(shape), dtype=tf.float32)
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regular)(w))
    return w


def get_b(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = get_w([2, 11], 0.01)
b1 = get_b([11])
y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = get_w([11, 1], 0.01)
b2 = get_b([1])
y = tf.matmul(y1, w2) + b2

loss_mse = tf.reduce_mean(tf.square(y - y_))
loss_total = loss_mse + tf.add_n(tf.get_collection("losses"))
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(40000):
        start = (i * BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = sess.run(y, feed_dict={x: grid})
    probs = probs.reshape(xx.shape)

plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[0.5])
plt.show()
