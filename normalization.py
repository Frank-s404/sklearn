# -*- codeing = utf-8 -*-
#@Time: 2021/11/17 16:31
#@Name: normalization.py

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets._samples_generator import make_classification #生成数据
from sklearn.svm import SVC #处理其中的一个model
import matplotlib.pyplot as plt #数据可视化

# a = np.array([[10, 2.3, 4.5],
#               [3234, 5, 0.3],
#               [23, 34.5, 2.2]], dtype = np.float64)
# #np.dtype:指定矩阵数据类型
# print(a) #用于做对比
# print(preprocessing.scale(a)) #标准化处理


X, y = make_classification(n_samples=300, n_features=2 , n_redundant=0, n_informative=2,
                           random_state=22, n_clusters_per_class=1, scale=100)
#生成data

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
#可视化，可见被明显地分成两类

X = preprocessing.scale(X)#normalization过程

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3) #X,Y是要测试的样本，.3是测试样本占比
clf = SVC() #分类
clf.fit(X_train, y_train)#训练模型

print(clf.score(X_test, y_test))#模型test结果打分

