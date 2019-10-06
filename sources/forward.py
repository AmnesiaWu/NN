import tensorflow as tf


def get_w(shape, regul):
    w = tf.Variable(tf.random.normal(shape), dtype=tf.float32)
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regul)(w))
    return w


def get_b(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b


def forward(x, regul):
    w1 = get_w([2, 11], regul)
    b1 = get_b([11])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_w([11, 1], regul)
    b2 = get_b([1])
    y = tf.matmul(y1, w2) + b2

    return y
