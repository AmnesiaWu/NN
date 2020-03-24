# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  

salary = [2500, 25.331233, 2700, 5600, 6700, 5400, 3100, 3500, 7600, 7800,
          8700, 9800, 10400]

group = numpy.arange(0, 11000, 1000)

print(salary)
plt.hist(salary, group, histtype='bar', rwidth=0.9)

plt.xlabel('salary-group')
plt.ylabel('salary')

plt.title(u'测试例子——直方图', FontProperties=font)

plt.show()