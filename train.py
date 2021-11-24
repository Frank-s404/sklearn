# -*- codeing = utf-8 -*-
#@Time: 2021/11/15 17:10
#@Name: train.py

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

print(iris_X[:3, :])#花的一些属性（3个数据集）
print(iris_y)#花的类别

X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, test_size=0.3)
#将所有的data分为测试类和学习类两类
#测试总数据，测试总数据，测试数据占比（y_test占总数据的30%）

knn = KNeighborsClassifier()#重命名
knn.fit(X_train, y_train)#自动完成圈点布置
print(knn.predict(X_test))#预测这是哪一种花
print(y_test)#实际结果（真实值）