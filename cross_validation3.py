# -*- codeing = utf-8 -*-
#@Time: 2021/11/20 17:11
#@Name: cross_validation3.py

from __future__ import print_function
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
X = digits.data
y = digits.target
param_range = np.logspace(-6, -2.3, 5) #创建等比数列
# 默认底数base = 10
# -6：生成数组的初始值为base的-6次方
# -2.3：生成数组的结束值为base的-2.3次方
# 5：总共生成5个数
train_loss, test_loss = validation_curve(
        SVC(), X, y, param_name='gamma', param_range=param_range, cv=10)
# mean_squared_error?
train_loss_mean = -np.mean(train_loss, axis=1) #求每一行的平均值，axis = 0求每一列的平均值，默认求所有的平均值
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(param_range, train_loss_mean,  color="r",
             label="Training")
plt.plot(param_range, test_loss_mean,  color="g",
             label="Cross-validation")

plt.xlabel("gamma") #体现每一个gamma的值
plt.ylabel("Loss")
plt.legend(loc="best") #加上图例
plt.show()