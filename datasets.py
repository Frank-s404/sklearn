# -*- codeing = utf-8 -*-
#@Time: 2021/11/16 16:00
#@Name: datasets.py

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
#线性回归
import matplotlib.pyplot as plt

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()#使用默https://www.lanqiao.cn/live/492/认值即可
model.fit(data_X, data_y)
#
# print(model.predict(data_X[:4, :]))
# print(data_y[:4])#取出整体的第四个元素

# X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=15)
#自己设定参数(接下来有参数注释)
#一个xdata和一种ydata的plot
# plt.scatter(X, y)#制图
# plt.show()#展示
print(model.coef_)#x乘上的值 y = kx + b,这里指的是k
print(model.intercept_)#与y轴的交点
print(model.get_params())#打印出之前定义好的参数及数值
print(model.score(data_X, data_y))#对模型学到的东西进行打分（吻不吻合）
#R^2 coefficient of determination
#有百分之多少的结果是正确的