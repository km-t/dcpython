import numpy as np

a = np.arange(9).reshape(3, 3)
b = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        b[i][j] = i+3*j
print(a)
print(b)
c = np.array(a[0:3, 0:1]-b[0:3, 0:1])
print(c)
