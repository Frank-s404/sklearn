# -*- codeing = utf-8 -*-
#@Time: 2021/11/21 16:22
#@Name: save.py
from sklearn import svm
from sklearn import datasets
import pickle
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X,y)

with open("models/clf.pickle", "wb") as f:
    pickle.dump(clf, f)