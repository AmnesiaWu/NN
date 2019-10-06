import mnist_backward
import mnist_forward
import tensorflow as tf
import numpy as np
from PIL import Image


def restore_model(test_pic):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [1, mnist_forward.Input_Node])
        y = mnist_forward.forward(x, None)
        pre_value = tf.argmax(y, 1)

        ema = tf.train.ExponentialMovingAverage(mnist_backward.AVERAGE_DECAY)
        ema_to_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                pre_value = sess.run(pre_value, feed_dict={x: test_pic})
                return pre_value
            else:
                print("No checkpoint path")
                return -1


def pre_pic(test_path):
    img = Image.open(test_path)
    re_img = img.resize((28, 28), Image.ANTIALIAS)#消除锯齿的方法处理图片
    im_arr = np.array(re_img.convert("L"))#变成灰度图
    std = 50
    for i in range(28):#反色
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if im_arr[i][j] < std:
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1/255)#将像素点变为0-1
    return img_ready


def application():
    test_num = eval(input("input the number of test pictures:"))
    for i in range(test_num):
        test_pic_path = input("input the path of test picture:")
        test_pic_ard = pre_pic(test_pic_path)
        pre_value = restore_model(test_pic_ard)
        print("Number:{}".format(pre_value))


def main():
    application()


if __name__ == "__main__":
    main()
