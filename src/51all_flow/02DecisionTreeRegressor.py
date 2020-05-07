import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = DecisionTreeRegressor()
model.fit(data_X, data_y)

housing_predict = model.predict(data_X)
tree_mse = mean_squared_error(data_y, housing_predict)
# tree_rmse为预测误差值
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)


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


# 结论：该模型训练的预测误差为0.0，交叉验证得到预测误差为5.77，上下浮动2.00，所以模型严重过拟合。
