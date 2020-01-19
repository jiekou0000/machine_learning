# 线性回归 -- 普通最小二乘法
#   计算 系数与截距 的问题   使得数据集实际观测数据和预测数据(估计值)之间的残差平方和最小
#
#  如果X是一个形状为 (n_samples, n_features)的矩阵，复杂度为 n_samples * (n_features * n_features)


import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


x_train = np.array([[0], [1], [2], [3], [4], [5]])
y_train = np.array([0, 1, 2, 3, 4, 5])

x_test = np.array([[6], [7], [8]])
y_test = np.array([6, 7, 8])

reg = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
reg.fit(x_train, y_train)

y_predict = reg.predict(x_test)


print('LR_score: ', reg.score(x_train, y_train))
print('LR_coef_: ', reg.coef_)
print('LR_intercept_: ', reg.intercept_)


# Plot outputs
plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, y_predict, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel("x")
plt.ylabel("y")

plt.show()