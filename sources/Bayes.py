# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
# p(L|X)  正比于 p(X|L) p(L) = p(xi|L) p(L) for i in range(len(X))

class Bayes:
    def __init__(self):
        self.labelsfre = {} # p(label)
        self.labels_vectors = {} # 每个label对应的数据向量

    def fit(self, dataset:list, labels:list):
        """
        训练数据
        :param dataset: 训练集
        :param labels: 标签
        :return: self
        """
        if len(dataset[0]) <= 0:
            raise ValueError(u"数据集维度错误")
        if len(dataset) != len(labels):
            raise ValueError(u'数据集和标签长度不匹配')
        nonlabels = set(labels)
        labels_num = len(labels)
        for item in nonlabels:
            self.labelsfre[item] = labels.count(item) / labels_num # 算出p(label)
        for vector, label in zip(dataset, labels):
            if label not in self.labels_vectors:
                self.labels_vectors[label] = []
            self.labels_vectors[label].append(vector) # 每个标签有多少个数据向量
        return self

    def predict(self, pre_data, classes):
        """
        对测试数据分类
        :param pre_data: 测试数据
        :param classes: 类型集合
        :return: 类型
        """
        probability = {}
        for thisclass in classes:
            p = 1
            labels_vectorsT = np.array(self.labels_vectors[thisclass]).T
            for xi in range(len(pre_data)):
                list_labels_vectorsT = list(labels_vectorsT[xi]) # list才有.count()
                p *= list_labels_vectorsT.count(pre_data[xi]) / len(self.labels_vectors[thisclass]) # p(xi|L)
            p *= self.labelsfre[thisclass]
            probability[thisclass] = p
        res = sorted(probability.items(), key=lambda item: -item[1])
        return res[0]

if __name__ == '__main__':
    data = pd.read_csv(r'D:\py_pro\test\data\泰坦尼克号数据.csv')
    for i in range(len(data)):
        if data['Age'].isnull()[i] :
            data['Age'][i] = data['Age'].mean()
        if data['Sex'][i] == 'male':
            data['Sex'][i] = 1
        else:
            data['Sex'][i] = 0
    train_data = data.iloc[:, 4:8].values
    labels = data.iloc[:, 1].values
    labels = list(labels)
    test_data = [1, 20, 0, 0]
    bayes = Bayes()
    bayes.fit(train_data, labels)
    res = bayes.predict(test_data, [0, 1])
    print(res)
