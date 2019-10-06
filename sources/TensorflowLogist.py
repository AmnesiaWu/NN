from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('data', one_hot=True)

trainimg = mnist.train.images
trainlable = mnist.train.labels
testimg = mnist.test.images
testlable = mnist.test.labels

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

acvt = tf.nn.softmax(tf.matmul(x, W) + b) # 模型结果 softmax 多分类的时候用
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(acvt), reduction_indices=1))# 逻辑回归损失函数
optm = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

pre = tf.equal(tf.argmax(acvt, 1), tf.argmax(y, 1))
accu = tf.reduce_mean(tf.cast(pre, 'float')) # 计算准确率 cast函数将pre转为float

Times = 50
batch_size = 100
display_step = 5
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(Times):
        cost_mean = 0.
        num_batch = int(mnist.train.num_examples / batch_size)
        for j in range(num_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optm, feed_dict={x : batch_x, y : batch_y})
            cost_mean += sess.run(cost, feed_dict={x : batch_x, y : batch_y}) / num_batch
        if i % display_step == 0:
            test_accu = sess.run(accu, feed_dict={x : testimg, y : testlable})
            print('cost:{}, accu:{}'.format(cost_mean, test_accu))