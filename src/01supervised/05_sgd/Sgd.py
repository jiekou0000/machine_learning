import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
y = np.array([1, 2, 3, 4])

clf = linear_model.SGDClassifier(max_iter=1000000, tol=0.001, warm_start=False)
clf.fit(X, y)


print(clf.predict([[2, 3]]))
print(clf.coef_)
print(clf.intercept_)


# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))

# Put the result into a color plot
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], color='red')
plt.show()


