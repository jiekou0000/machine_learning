# 多项分布朴素贝叶斯 demo   极大似然估计  
#
# 给出训练集
# Xa  1  1  1  1  1  2  2  2  2  2  3  3  3  3  3
# Xb  1  2  2  1  1  1  2  2  3  3  3  2  2  3  3
# Y  -1 -1  1  1 -1 -1 -1  1  1  1  1  1  1  1 -1
# 计算(2, 1)分别属于1，-1的概率  -->  1/45, 1/15

# P(Y=1) = 9/15, P(Y=-1) = 6/15
# P(Xa=1|Y=1) = 2/9, P(Xa=2|Y=1) = 3/9, P(Xa=3|Y=1) = 4/9,   P(Xb=1|Y=1) = 1/9, P(Xb=2|Y=1) = 4/9, P(Xb=3|Y=1) = 4/9
# P(Xa=1|Y=-1) = 3/6, P(Xa=2|Y=-1) = 2/6, P(Xa=3|Y=-1) = 1/6,   P(Xb=1|Y=-1) = 3/6, P(Xb=2|Y=-1) = 2/6, P(Xb=3|Y=-1) = 1/6
# P(Y=1)P(Xa=2|Y=1)P(Xb=1|Y=1) = 9/15 * 3/9 * 1/9 = 1/45 = 0.022
# P(Y=-1)P(Xa=2|Y=-1)P(Xb=1|Y=-1) = 6/15 * 2/6 * 3/6 = 1/15 = 0.066


import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt

X = np.array([[1, 1], [1, 2], [1, 2], [1, 1], [1, 1], [2, 1], [2, 2], [2, 2], [2, 3], [2, 3], [3, 3], [3, 2], [3, 2], [3, 3], [3, 3]])
Y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])

# clf = MultinomialNB(alpha=1.0e-10, fit_prior=True, class_prior=None)
clf = MultinomialNB()
clf.fit(X, Y)


print(clf.predict_proba([[2, 1]]))
print(clf.score([[2, 1]], [-1]))
print(clf.score([[2, 1]], [1]))
print(clf.predict([[2, 1]]))

y_pred = clf.fit(X, Y).predict(X)
print("Number of mislabeled points out of a total %d points : %d" % (X.shape[0],(Y != y_pred).sum()))
print(Y == y_pred)
error = (Y != y_pred).sum()
print('the total number of errors: %d' % error)
print('the total error rate: %.4f' % (error/float(X.shape[0])))

# plt.scatter(X[:, 0], X[:, 1], c=Y)
metrics.plot_confusion_matrix(clf, X, Y)
plt.show()
