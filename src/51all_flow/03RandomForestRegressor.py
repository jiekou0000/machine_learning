import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = RandomForestRegressor()
model.fit(data_X, data_y)

housing_predict = model.predict(data_X)
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


# 保存模型
joblib.dump(model, "03RandomForestRegressorModel.pkl")
# and later...
my_model_loaded = joblib.load("03RandomForestRegressorModel.pkl")