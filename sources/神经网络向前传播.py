import tensorflow as tf
x = tf.placeholder(tf.float32, shape = (None, 2))
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(y, feed_dict={x: [[0.7, 0.5], [0.2, 0.3], [0.4, 0.2]]}))
    print(sess.run(w1))
    print(sess.run(w2))