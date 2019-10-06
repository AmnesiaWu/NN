import mnist_forward
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import mnist_generates
BATCH_SIZE = 200
RATE_DECAY = 0.99
RATE_BASE = 0.1
REGUL = 0.0001
STEPS = 50000
AVERAGE_DECAY = 0.99
MODEL_PATH = "./model/"
MODEL_NAME = "minist_model"
train_num_examples = 60000


def backward():
    x = tf.placeholder(tf.float32, [None, mnist_forward.Input_Node])
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.Output_Node])
    y = mnist_forward.forward(x, REGUL)
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection("losses"))

    learning_rate = tf.train.exponential_decay(RATE_BASE, global_step, train_num_examples / BATCH_SIZE, RATE_DECAY, staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    ema = tf.train.ExponentialMovingAverage(AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name="train")

    saver = tf.train.Saver()
    img_batch, label_batch = mnist_generates.get_tfRecord(BATCH_SIZE, True)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)#断电时保存

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)#线程协调器加快速率

        for i in range(STEPS):
            xs, ys = sess.run([img_batch, label_batch])#mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
            sess.run(global_step, feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("loss: {}".format(sess.run(loss,feed_dict={x: xs, y_: ys})))
                saver.save(sess, os.path.join(MODEL_PATH, MODEL_NAME), global_step=global_step)
        coord.request_stop()
        coord.join(threads)#关闭线程协调器

def main():
    #mnist = input_data.read_data_sets("./data/", one_hot=True)
    backward() # mnist


if __name__ == "__main__":
    main()