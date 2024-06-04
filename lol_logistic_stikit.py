import numpy as np
import matplotlib.pyplot as plt
file = open('0825_jug_kda_win.csv', 'r')

lines = file.readlines()
x = []; y = []
for j, line in enumerate(lines):
    if j == 0:
        line = line.rstrip('\n')
        x_label = line.split(',')[0]
        y_label = line.split(',')[1]
    else:
        line = line.rstrip('\n')
        x.append(float(line.split(',')[0]))
        y.append(float(line.split(',')[1]))
x = np.array(x)
y = np.array(y)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

x = x.reshape(-1, 1)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x, y)
w = model.coef_[0]
b = model.intercept_
print(w, b)

win_lose_vec = np.zeros(len(x))
for j in range(len(y)):
    y_hat = sigmoid(w * x[j] + b)
    win_lose = 'lose'
    if y_hat >= 0.5:
        win_lose = "win"
        win_lose_vec[j] = 1
    print(f'KAD :{x[j]}, WIN/LOSE : {y[j]}, MLR : {y_hat}, PRED : {win_lose}')
accuracy = np.sum(y == win_lose_vec) / len(y)
print(accuracy, len(y))