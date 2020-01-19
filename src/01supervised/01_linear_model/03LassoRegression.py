# Lasso是拟合稀疏系数的线性模型
# L1正则化


import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt


x_train = np.array([[0], [1], [2], [3], [4], [5]])
y_train = np.array([0, 1, 2, 3, 4, 5])

x_test = np.array([[6], [7], [8]])
y_test = np.array([6, 7, 8])


reg = Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
reg.fit(x_train, y_train)

y_predict = reg.predict(x_test)


print('Lasso_score: ', reg.score(x_train, y_train))
print('Lasso_coef_: ', reg.coef_)
print('Lasso_intercept_: ', reg.intercept_)


# Plot outputs
plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, y_predict, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


