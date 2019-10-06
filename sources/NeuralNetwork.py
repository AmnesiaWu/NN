import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score #计算准确率
from sklearn.model_selection import train_test_split #
mnist = input_data.read_data_sets('../data', one_hot=True)
trainimg = mnist.train.images
trainlable = mnist.train.labels
testimg = mnist.test.images
testlable = mnist.test.labels

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])

weights = {
    'w1':tf.Variable(tf.random_normal([784, 256], stddev=0.1)),
    'w2':tf.Variable(tf.random_normal([256, 128], stddev=0.1)),
    'out':tf.Variable(tf.random_normal([128, 10], stddev=0.1))
}
biases = {
    'b1':tf.Variable(tf.random_normal([256])),
    'b2':tf.Variable(tf.random_normal([128])),
    'out':tf.Variable(tf.random_normal([10]))
}


def multlayer(x, weights, biases):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['w1']), biases['b1']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['w2']), biases['b2']))
    res = tf.add(tf.matmul(layer2, weights['out']), biases['out'])
    return res

pre = multlayer(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pre, labels=y))
optm = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
corr = tf.equal(tf.argmax(pre, 1), tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(corr, 'float'))

#acc2 = accuracy_score(y, pre)
#x_trian, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
epoch = 500
batch_size = 100
dispaly_step = 5
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    total_cost = 0.
    for i in range(epoch):
        num = int(mnist.train.num_examples / batch_size)
        for j in range(num):
            train_x, train_y = mnist.train.next_batch(batch_size)
            feed = {x: train_x, y:train_y}
            sess.run(optm, feed_dict=feed)
            total_cost += sess.run(cost, feed_dict=feed)
        total_cost /= num
        if i % dispaly_step == 0:
            print(r'step:{}/{}, cost:{}, train_acc:{}, test_acc:{}'.format(i, epoch, total_cost, sess.run(acc2, feed_dict={x:trainimg, y:trainlable}), sess.run(acc, feed_dict={x:testimg, y:testlable})))