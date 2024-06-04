import numpy as np
import matplotlib.pyplot as plt
file = open('0825_jug_kda_win_100.csv', 'r')

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

a1 = np.sum(x*x)
b1 = np.sum(x)
c1 = b1
d1 = len(x)


z1 = np.sum(x*y)
z2 = np.sum(y)

m = 1.0 / (a1*d1 - b1*c1)
a = ( d1*z1 - b1*z2 ) * m
b = (-c1*z1 + a1*z2 ) * m
y_hat = a * x + b

loss = np.sum((y - y_hat) ** 2) / len(x)
print(f"Loss : {loss}")

plt.plot(x,y,'.')
plt.plot(x, y_hat, '-')
plt.show()