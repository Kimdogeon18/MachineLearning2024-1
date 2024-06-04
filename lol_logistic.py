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

def loss_ce(y, y_hat):
    eps = 1e-6
    return-np.mean(y*np.log(y_hat+eps) + (1-y)*np.log(1-y_hat+eps))

def gradient(x, y, y_hat):
    #dw = d L_ce(sigmoid(w*x + b), y) / d w
    dw = (y_hat - y) * x
    #db = d L_ce(sigmoid(w*x + b), y) / d b
    db = y_hat - y
    return dw, db

eta = 0.0001
epoch = 30

w, b = 1000, -50

for e in range(epoch):
    for j in range(len(x)):
        z = w * x[j] + b
        y_hat = sigmoid(z)
        dw, db = gradient(x[j], y[j], y_hat)
        w = w - eta * dw
        b = b - eta * db
    if e % 10 == 0: print(f'Epoch {e} : Lose = {loss_ce(y, sigmoid(w*x + b))}')

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