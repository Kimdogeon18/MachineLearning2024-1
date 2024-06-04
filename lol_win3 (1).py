import pandas as pd

df = pd.read_pickle("DATA/0825_game_data_1000.pkl")

print(df)

# 쉽게 정보접근
idx = pd.IndexSlice
# 집합 중복 제거
ids = set()
for id, team_id, feature in df.index:
    ids.add(id)
ids = list(ids)

x = []
y = []

# index, values
for j, id in enumerate(ids):
    # 보이는대로 빈칸에 값을 채워준다. 그래서 쉽게 데이터에 접근할 수 있다...
    # df(df. (df. (df>)))))
    blue_kda = df.loc[idx[id, 100, 'kda']].values
    red_kda = df.loc[idx[id, 200, 'kda']].values
    blue_win = df.loc[idx[id, 100, 'win']].values[0]
    #[[blue_kda(TOP, JUG,MID..) + red_kda(TOP., JUG, MID...)]]
    x.append(list(blue_kda) + list(red_kda))
    y.append(blue_win)

import numpy as np
x = np.array(x)
y = np.array(y)

print(x)
print(y)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR

# model = LinearRegression()
model = LogisticRegression()
model.fit(x, y)

print(model.coef_)
print(model.intercept_)

y_pred = model.predict(x)

predict = []
for pred in y_pred:
    if pred >= 0.5 : predict.append(1)
    else : predict.append(0)

predict = np.array(predict)
exact = y
accuracy = np.sum(predict == exact) / len(predict)
print("Accuracy : ", accuracy)