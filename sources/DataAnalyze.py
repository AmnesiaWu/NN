# -*- coding:utf8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import jieba
import jieba.analyse

data = pd.read_csv(r'../data/泰坦尼克号数据.csv')
cnt = 0
for i in range(len(data)):
    if data['Age'].isnull()[i]:
        data['Age'][i] = data['Age'].mean()
# data_t = data.values.T
# age = data_t[5]
# age_rg = age.max() - age.min()
# age_width = age_rg / 12
# bi = np.arange(0, 90, 10)
# print(type(age))
# plt.hist(list(age), bi, rwidth=0.9)
# plt.show()

# arr = np.array([[1, 2, 3], [8, 10, 15]])
# arr2 = np.array([[45, 22, 5], [25, 65, 44]])
# test = np.concatenate((arr, arr2)) # 数据整合
# print(test)

# 离差标准化——消除单位影响以及变异大小因素的影响
# new_data = (data - min) / (max - min)

# 零-均值标准化(标准差标准化)
# new_data = (data - mean) / std

# 小数定标规范化
# k = np.ceil(np.log10(data.abs().max()) # np.ceil(3.1) = 4
# new_data = data / 10**k

# 等宽离散化
# age = data['Age'].T.values
# sorted_class = pd.cut(age, 4, labels=['童年', '年轻', '中年', '老年'])
# print(sorted_class)

data2 = open(r'D:\py_pro\test\data\test.txt').read()
res = jieba.cut(data2)# 关键词提取
print(res)