# -*- coding: UTF-8 -*-
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sources.mnist_forward
import sources.mnist_backward
import sources.mnist_generates
TEST_NUM = 10000


def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, sources.mnist_forward.Input_Node])
        y_ = tf.placeholder(tf.float32, [None, sources.mnist_forward.Output_Node])
        y = sources.mnist_forward.forward(x, None)

        ema = tf.train.ExponentialMovingAverage(sources.mnist_backward.AVERAGE_DECAY)#实例化滑动平均
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        img_batch, label_batch = sources.mnist_generates.get_tfRecord(TEST_NUM, False)#

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(sources.mnist_backward.MODEL_PATH)#把滑动平均值赋给参数
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)#恢复模型
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 线程协调器加快速率

                    xs, ys = sess.run([img_batch, label_batch])##
                    accuracy_score = sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print("step:{} accuracy: {}".format(global_step, accuracy_score))

                    coord.request_stop()
                    coord.join(threads)  # 关闭线程协调器
                else:
                    print("Can't find")
            time.sleep(5)
def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist)

if __name__ == "__main__":
    main()