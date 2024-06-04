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

#x1,x2,y 데이터 가공
red_jug_kda = []
blue_jug_kda = []
red_win = []
for j in range(len(x)):
    if j % 2 == 0:
        red_jug_kda.append(x[j])
        red_win.append(y[j])
    else:
        blue_jug_kda.append(x[j])


for j in range(len(red_win)):
    print(j, red_jug_kda[j], blue_jug_kda[j], red_win[j])

x1 = np.array(red_jug_kda)
x2 = np.array(blue_jug_kda)
y = np.array(red_win)

#y = a*x1 + b*x2 +c
#? a, b, c
#A = [ [a1,a2,a3],
#       [a4,a5,a6],
#       [a7,a8,a9] ]
# y_vec = [ [y1],
#           [y2],
#           [y3] ]
a1 = np.sum(x1*x1); a2 = np.sum(x1*x2); a3 = np.sum(x1)
a4 = a2;            a5 = np.sum(x2*x2); a6 = np.sum(x2)
a7 = a3;            a8 = a6;            a9 = len(x1)

A = np.array([ [a1,a2,a3],
               [a4,a5,a6],
               [a7,a8,a9] ])

y1 = np.sum(x1*y)
y2 = np.sum(x2*y)
y3 = np.sum(y)

y_vec = np.array([y1, y2, y3])

param= np.linalg.solve(A,y_vec)
a = param[0]
b = param[1]
c = param[2]
win_lose_vec = np.zeros(len(x1))
for j in range(len(y)):
    y_hat = a * x1[j] + b * x2[j] + c
    win_lose = 'lose'
    if y_hat >= 0.5 :
      win_lose = "win"
      win_lose_vec[j] = 1
    print(f'RED KAD :{x1[j]}, BLUE KDA: {x2[j]}, WIN/LOSE : {y[j]}, MLR : {y_hat}, PRED : {win_lose}')
accuracy = np.sum(y == win_lose_vec) / len(y)
print(accuracy)






 # print(A)
 # print(a,b,c)
 # print(y_vec)
#
# fig =plt.figure()
# ax = fig.add_subplot(projection='3d')
#
# ax.scatter(x1,x2,y)
# ax.set_xlabel('RED JUG KDA')
# ax.set_ylabel('BLUE JUG KDA')
# ax.set_zlabel('WIN/LOSE')
# plt.show()