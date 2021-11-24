# -*- codeing = utf-8 -*-
#@Time: 2021/11/18 15:24
#@Name: cross_validation.py
#对比各种属性（参数）看哪个更好
#用test_data来对比我们的训练结果

from __future__ import print_function
#引用最新版本的特性
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np


iris = load_iris() #下载iris花模型
X = iris.data
y = iris.target

# test train split #
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 4) #分开test和training
knn = KNeighborsClassifier(n_neighbors=5) #训练机器学习 数据点周围5个值
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(y_pred)
print(knn.score(X_test, y_test))

#cross_validation已经被替换掉
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy') #cv:选择折叠的次数(自动分成五组（不一样的training和test data）) accuracy:评价指标是准确度

print(scores)#z
print(scores.mean())#求平均，综合5个的结果

import matplotlib.pyplot as plt
k_range = range(1, 31)#定义n_neighbors哪个参数比较好
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
##    loss = -cross_val_score(knn, X, y, cv=10, scoring='mean_squared_error') # for regression 线性回归（生成一个负值）
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy') # for classification
    k_scores.append(scores.mean())#在score内部加入每个打分结果
#精确度见图
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
