import tensorflow as tf
import forward
import generater
import numpy as np
import matplotlib.pyplot as plt

step = 40000
batch_size = 30
base = 0.001
decay = 0.999


def backward():
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))
    X, Y_, Y_c = generater.generater()
    y = forward.forward(x, 0.01)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(base, global_step, 300/batch_size, decay, staircase=True)

    loss_mse = tf.reduce_mean(tf.square(y-y_))
    loss_total = loss_mse + tf.add_n(tf.get_collection("losses"))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(step):
            start = (i * batch_size) % 300
            end = start + batch_size
            sess.run(train_step, feed_dict={x: X[start: end], y_: Y_[start: end]})
            if i % 2000 == 0:
                print("loss:{}".format(sess.run(loss_total, feed_dict={x: X, y_: Y_})))
        xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = sess.run(y, feed_dict={x: grid})
        probs = probs.reshape(xx.shape)

    plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
    plt.contour(xx, yy, probs, levels=[0.5])
    plt.show()


if __name__ == '__main__':
    backward()
