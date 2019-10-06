FILE_OBJECT = open('order.log', 'r', encoding='UTF-8')
import tensorflow as tf
BASE = 0.1
DECAY = 0.99
STEP = 1
# wn+1 = wn - learning_rate * (损失函数的梯度(导数))
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(BASE, global_step, STEP, DECAY, staircase=True)

w = tf.Variable(tf.constant(5, dtype=tf.float32))

loss = tf.square(w + 1)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(40):
        sess.run(train_step)
        print("{}次， w:{}, rate:{}, loss:{},step:{}".format(i, sess.run(w), sess.run(learning_rate), sess.run(loss), sess.run(global_step)))
