# -*- coding:utf-8 -*-
import numpy as np
from collections import defaultdict

def knn(k, train_data, test_data, labels):
    """
    :param k: 取k个点判断
    :param train_data:训练集，array
    :param test_data:测试数据，array
    :param labels:训练数据标签
    :return:test_data的分类结果
    """
    data_size = len(train_data)
    differ = np.tile(test_data, (data_size, 1)) - train_data
    squ = differ ** 2
    distances = np.sqrt(squ.sum(axis=1))
    index_distance = distances.argsort()
    cnt = defaultdict(int)
    for i in range(k):
        cnt[labels[index_distance[i]]] += 1
    res = sorted(cnt.items(), key=lambda item:-item[1])# 取频数最大的label
    return res[0][0]
