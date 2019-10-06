from PIL import Image as ig
import numpy as np

array_L = np.array(ig.open(r'C:\Users\Administrator\Pictures\Camera Roll\宇宙\timg (8).jfif').convert('L')).astype('float')
depth = 10 # 0-100
grad_x, grad_y = np.gradient(array_L)   #取图像灰度的梯度值
grad_x = grad_x * depth / 100   #归一化
grad_y = grad_y * depth / 100   #归一化
temp = np.sqrt(grad_y ** 2 + grad_x ** 2 + 1.)
uni_x = grad_x / temp
uni_y = grad_y / temp
uni_z = 1. / temp

vec_el = np.pi / 2.2    #光源俯视角度，弧度
vec_az = np.pi / 4.     #光源方向角度，弧度
dx = np.cos(vec_el) * np.cos(vec_az)    #光源对x的影响
dy = np.cos(vec_el) * np.sin(vec_az)    #对y...
dz = np.sin(vec_el)                     #对z...

new_array = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)    #光源归一化
new_array = new_array.clip(0, 255)

im = ig.fromarray(new_array.astype('uint8'))
im.show()
