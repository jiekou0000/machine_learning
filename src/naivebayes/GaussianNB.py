# 高斯朴素贝叶斯  以iris为demo，共150条数据 前148条作为训练集，测试后2条数据
# iris中有6条数据不属于高斯模型

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
gnb = GaussianNB()

# y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
# print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))
# print(iris.target == y_pred)  # 倒数第17个点[6.3, 2.8, 5.1, 1.5, 2]不在模型中
# error = (iris.target != y_pred).sum()
# print('the total number of errors: %d' % error)
# print('the total error rate: %.4f' % (error/float(iris.data.shape[0])))

train_data = iris.data[:148]
train_target = iris.target[:148]
gnb.fit(train_data, train_target)


print(gnb.predict([[6.2, 3.4, 5.4, 2.3]]))
print(gnb.predict([[5.9, 3.0, 5.1, 1.8]]))


print(gnb.score([[6.3, 2.8, 5.1, 1.5]], [0]))
print(gnb.score([[6.3, 2.8, 5.1, 1.5]], [1]))
print(gnb.score([[6.3, 2.8, 5.1, 1.5]], [2]))
print(gnb.predict([[6.3, 2.8, 5.1, 1.5]]))



