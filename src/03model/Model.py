# test


import numpy as np
from sklearn.model_selection import KFold


X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
rkf = KFold(n_splits=2)

for train_index, test_index in rkf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    # 创建训练/测试集合:
    # X_train, X_test = X[train_index], X[test_index]
    # y_train, y_test = y[train_index], y[test_index]
    # print(X_train, X_test, y_train, y_test)

