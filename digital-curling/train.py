import random as rd
import numpy as np
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import time
from chainer import Variable, optimizers, Chain, serializers
import pandas as pd


df_origin = pd.read_csv('./logs.csv', sep=',')
df = df_origin.sample(frac=1)
inputData = np.empty((0, 14), dtype=np.float32)
result = np.empty((0, 1), dtype=np.float32)
dSize = 4659691-2
trainSize = int(dSize/1000)
testSize = dSize-trainSize
# for i in range(trainSize):
for i in range(50):
    vec = str(df.iloc[i, 0])
    v = np.zeros(14, dtype=np.float32)
    for j in range(11):
        v[j] = float(vec[j+1])
    print(v)
    v[11] = df.iloc[i, 1]
    v[12] = df.iloc[i, 2]
    v[13] = df.iloc[i, 3]
    hoge = np.array([v], dtype=np.float32)
    inputData = np.append(inputData, hoge, axis=0)
    r = np.array([[float(df.iloc[i, 4])]], dtype=np.float32)
    result = np.append(result, r, axis=0)
x = Variable(inputData)
t = Variable(result)


class Model(Chain):
    def __init__(self):
        super(Model, self).__init__(

            l1=L.Linear(14, 100),
            l2=L.Linear(100, 100),
            l3=L.Linear(100, 1),
        )

    def __call__(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        return h3


model = Model()
optimizer = optimizers.Adam()
optimizer.setup(model)

lossData = []
for i in tqdm(range(trainSize)):
    model.zerograds()
    y = model(x)
    loss = F.mean_squared_error(y, t)
    loss.backward()
    optimizer.update()
    lossData.append(loss.data)


plt.plot(lossData)
plt.xlabel("No of trails")
plt.ylabel("loss")
plt.title("or.py")
plt.legend()


plt.show()
