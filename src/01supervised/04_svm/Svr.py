import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 2, 3, 4, 5])

clf = SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.2, shrinking=True, cache_size=500, verbose=False, max_iter=-1)
clf.fit(X, y)
y_predict = clf.predict(X)


print(clf.predict([[6]]))


# Plot outputs
plt.scatter(X, y, color='black')
plt.plot(X, y_predict, color='blue', linewidth=3)
plt.show()
