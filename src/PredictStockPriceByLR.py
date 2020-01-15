# 预测股票价格

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 从文件中获取数据
origDf = pd.read_csv('D:/stockData/ch13/6035052018-09-012019-05-31.csv', encoding='gbk')
df = origDf[['Close', 'High', 'Low', 'Open', 'Volume']]
featureData = df[['Open', 'High', 'Volume', 'Low']]

# 划分特征值和目标值
feature = featureData.values
target = np.array(df['Close'])