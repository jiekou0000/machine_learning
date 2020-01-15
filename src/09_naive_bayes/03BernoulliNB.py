# 伯努利朴素贝叶斯 demo
#
# 给出训练集 -- 特征变量只有二元
# Xa  0  0  1  1  1  0  0  1  1  1  0  1  1  1  0
# Xb  0  0  1  1  1  1  1  1  1  0  0  1  1  0  0
# Y  -1 -1  1  1 -1 -1 -1  1  1  1  1  1  1  1 -1
# 计算(0, 1)分别属于1，-1的概率


import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
import matplotlib.pyplot as plt

X = np.array([[0, 0], [0, 0], [1, 1], [1, 1], [1, 1], [0, 1], [0, 1], [1, 1], [1, 1], [1, 0], [0, 0], [1, 1], [1, 1], [1, 0], [0, 0]])
Y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])

clf = BernoulliNB()
clf.fit(X, Y)


print(clf.predict_proba([[0, 1]]))
print(clf.score([[0, 1]], [-1]))
print(clf.score([[0, 1]], [1]))
print(clf.predict([[0, 1]]))

y_pred = clf.fit(X, Y).predict(X)
print("Number of mislabeled points out of a total %d points : %d" % (X.shape[0],(Y != y_pred).sum()))
print(Y == y_pred)
error = (Y != y_pred).sum()
print('the total number of errors: %d' % error)
print('the total error rate: %.4f' % (error/float(X.shape[0])))

metrics.plot_confusion_matrix(clf, X, Y)
plt.show()
