#-*- coding:utf8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold #交叉验证
from sklearn import model_selection#交叉验证
from sklearn.ensemble import RandomForestClassifier #随机森林
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif #特征选择
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)#pandas显示所有行列
titanic = pd.read_csv("泰坦尼克号数据.csv")

titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
titanic['Embarked'] = titanic['Embarked'].fillna('S')#填充缺少的值

titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1#将性别特征改为0，1识别

titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0
titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2#将目的地改为INT类型方便识别

#
# 线性回归或逻辑
# alg = LinearRegression()
#kf = KFold(n_splits=3,random_state=1)
# Predictions = []
# for train, test in kf.split(titanic):
#     train_predictors = titanic[predictors].iloc[train]
#     train_target = titanic['Survived'].iloc[train]
#     alg.fit(train_predictors, train_target)
#     Predictions.append(alg.predict(titanic[predictors].iloc[test]))
# Predictions = np.concatenate(Predictions, axis=0)
# Predictions[Predictions > .5] = 1
# Predictions[Predictions <= .5] = 0
# accu = sum(Predictions == titanic['Survived']) / len(Predictions)
# print(accu)

# 线性回归或逻辑

# 随机森林
titanic['Num'] = titanic['SibSp'] + titanic['Parch']
titanic['Length'] = titanic['Name'].apply(lambda x:len(x))
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Num', 'Length']
# alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=5, min_samples_leaf=1)# 随机森林
# scores = model_selection.cross_val_score(alg, titanic[predictors], titanic['Survived'], cv = 5)# 交叉验证简单方法，返回3次交叉验证的结果
# print(scores.mean())
# 随机森林

# 特征选择
selector = SelectKBest(f_classif, k = 8)
selector.fit(titanic[predictors], titanic['Survived'])
scores = -np.log10(selector.pvalues_)
plt.bar(predictors, scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical') # 改变X坐标
plt.show()
# 特征选择