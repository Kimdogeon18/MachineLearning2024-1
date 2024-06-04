import pandas as pd

df = pd.read_pickle('0825_game_data_1000.pkl')

idx = pd.IndexSlice
ids = set()
for id, team_id, feature in df.index: ids.add(id)
ids = list(ids)

predict = []
exact = []
for j, id in enumerate(ids):
    # if j == 10: break
    blue_kda = df.loc[idx[id, 100, 'kda']].values
    red_kda = df.loc[idx[id, 200, 'kda']].values
    blue_win = df.loc[idx[id, 100, 'win']].values[0]
    exact.append(blue_win)
    pred = 0
    if sum(blue_kda) >= sum(red_kda):
        pred = 1
    predict.append(pred)
    print(sum(blue_kda),  sum(red_kda), blue_win, pred)

import numpy as np
predict = np.array(predict)
exact = np.array(exact)
accuary = np.sum(predict == exact) / len(predict)
print("Accuracy : ", accuary)

