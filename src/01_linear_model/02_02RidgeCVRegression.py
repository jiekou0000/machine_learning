# 广义交叉验证的Ridge
#  自己得出给定的alphas当中的最适合的alpha值


import numpy as np
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt


x_train = np.array([[0], [1], [2], [3], [4], [5]])
y_train = np.array([0, 1, 2, 3, 4, 5])

x_test = np.array([[6], [7], [8]])
y_test = np.array([6, 7, 8])

reg = RidgeCV(alphas=[1e-2, 1e-1, 1.0, 10.0], fit_intercept=True, normalize=False, scoring=None, cv=None, gcv_mode=None, store_cv_values=False)
reg.fit(x_train, y_train)

y_predict = reg.predict(x_test)


print('RCV_alpha_: ', reg.alpha_)
print('RCV_score: ', reg.score(x_train, y_train))
print('RCV_coef_: ', reg.coef_)
print('RCV_intercept_: ', reg.intercept_)


# Plot outputs
plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, y_predict, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
