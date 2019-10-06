import tensorflow as tf
Input_Node = 784
Output_Node = 10
Layer_Node = 500


def get_w(shape, regul):
    w = tf.Variable(tf.random_normal(shape, stddev=0.1))
    if regul != None :
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regul)(w))
    return w


def get_b(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def forward(x, regul):
    w1 = get_w([Input_Node, Layer_Node], regul)
    b1 = get_b([Layer_Node])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_w([Layer_Node, Output_Node], regul)
    b2 = get_b([Output_Node])
    y = tf.matmul(y1, w2) + b2
    return y
