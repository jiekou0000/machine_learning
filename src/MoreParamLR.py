# 以波士顿房价数据为案例，实现基于多个特征值的线性回归

from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 加载数据
dataset = datasets.load_boston()

# 特征值集合，不包括目标值房价
featureData = dataset.data
housePrice = dataset.target

lrTool = LinearRegression()
lrTool.fit(featureData, housePrice)

# 输出系数和截距
print(lrTool.coef_)
print(lrTool.intercept_)

# 画图显示
plt.scatter(housePrice, housePrice, label='Real Data')
plt.scatter(housePrice, lrTool.predict(featureData), c='R', label='Predicted Data')

plt.legend(loc='best')  # 绘制图例
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel("House Price")
plt.ylabel("Predicted Price")
plt.show()




