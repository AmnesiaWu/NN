from PIL import Image
import os
import glob
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

path = r'D:\py_pro\test\pic\captcha_test'
path = os.path.join(path, '*g')
files = glob.glob(path)
num_files = len(files)
counter_correct = 0
sess = tf.Session()
saver = tf.train.import_meta_graph(r'../Captcha_model/captcha.ckpt-246200.meta')
saver.restore(sess, r'../Captcha_model/captcha.ckpt-246200')
graph = tf.get_default_graph()
y_pre = graph.get_tensor_by_name("output/y_pre:0")
x = graph.get_tensor_by_name("input/x:0")
keep_pro = graph.get_tensor_by_name('keep_pro/keep_pro:0')
for file in files:
    image = Image.open(file)
    image = np.array(image.convert('L'))
    image = image.flatten()
    image = image.astype(np.float32)
    image = np.multiply(image, 1. / 255.)
    image = image.reshape(1, 160 * 60)
    label = os.path.basename(file)
    label = label[:4]
    res = sess.run(y_pre, feed_dict={x: image, keep_pro: 1.})
    res = res.reshape([-1, 4, 10])
    pre = res
    pre = pre.argmax(2)
    st = [st for st in pre[0]]
    pre_str = ''.join(str(s) for s in st)
    if pre_str == label:
        counter_correct += 1
        print('label:{}, pre:{}, 预测正确'.format(label, pre_str))
    else:
        print('label:{}, pre:{}, 预测错误'.format(label, pre_str))
print("准确率:{}".format(counter_correct / num_files))