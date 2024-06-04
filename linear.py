import numpy as np
import matplotlib.pyplot as plt

a = 5,12
b = 14,45

x = np.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([10, 20, 25, 30, 40, 45, 40, 50, 60, 55])

A = np.array()
a1 = np.sum(x*x)
b1 = np.sum(x)
c1 = b1
d1 = len(x)

A = np.array()
#
# z1 = np.sum(x*y)
# z2 = np.sum(y)
#
# m = 1.0 / (a1*d1 - b1*c1)
# a = ( d1*z1 - b1*z2 ) * m
# b = (-c1*z1 + a1*z2 ) * m
#
y_hat = a * x + b

loss = np.sum((y - y_hat) ** 2) / len(x)
print(f"Loss : {loss}")

plt.plot(x, y, '.')
plt.plot(x, y_hat, '-')
plt.show()