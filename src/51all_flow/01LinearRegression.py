import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# 使用以后的数据集进行线性回归（这里是波士顿房价数据）
loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X, data_y)

print(model.predict(data_X[:4, :]))
print(data_y[:4])

housing_predict = model.predict(data_X)
# MSE:均方误差 RMSE:均方根误差
line_mse = mean_squared_error(data_y, housing_predict)
# line_rmse为预测误差值
line_rmse = np.sqrt(line_mse)
print(line_rmse)


# 交叉验证
scores = cross_val_score(model, data_X, data_y, scoring='neg_mean_squared_error', cv=10)
rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    # 得分
    print("Mean:", scores.mean())
    # 浮动值
    print("Standard deviation:", scores.std())

display_scores(rmse_scores)
