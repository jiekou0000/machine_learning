# ridge回归
# L2正则化， 使得 带罚项的残差平方和最小
# alpha越小，越会过拟合


import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt


x_train = np.array([[0], [1], [2], [3], [4], [5]])
y_train = np.array([0, 1, 2, 3, 4, 5])

x_test = np.array([[6], [7], [8]])
y_test = np.array([6, 7, 8])

reg = Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)
reg.fit(x_train, y_train)

y_predict = reg.predict(x_test)


print('Ridge_score: ', reg.score(x_train, y_train))
print('Ridge_coef_: ', reg.coef_)
print('Ridge_intercept_: ', reg.intercept_)


# Plot outputs
plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, y_predict, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


