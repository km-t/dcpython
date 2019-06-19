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
from chainer.training import extensions
import pandas as pd


class Model(Chain):
    def __init__(self):
        super(Model, self).__init__(

            l1=L.Linear(11, 100),
            l2=L.Linear(100, 200),
            l3=L.Linear(200, 100),
            l4=L.Linear(100, 2),
        )

    def __call__(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        h4 = self.l4(h3)
        return h4


def train(keyNum):
    inputData = np.empty((0, 11), dtype=np.float32)
    outputData = np.empty((0, 2), dtype=np.float32)
    for i in range(trainSize):
        vec = str(df_train.iloc[i, 0])
        v = np.zeros(11, dtype=np.float32)
        for j in range(11):
            v[j] = float(vec[j+1])
        IN = np.array([v], dtype=np.float32)
        o = np.zeros(2, dtype=np.float32)
        o[0] = df_train.iloc[i, keyNum]
        o[1] = df_train.iloc[i, 4]
        OUT = np.array([o], dtype=np.float32)
        inputData = np.append(inputData, IN, axis=0)
        outputData = np.append(outputData, OUT, axis=0)
    x = Variable(inputData)
    t = Variable(outputData)

    m = Model()
    optimizer = optimizers.Adam()
    optimizer.setup(m)

    lossData = []

    for i in tqdm(range(trainSize)):
        m.zerograds()
        y = m(x)
        loss = F.mean_squared_error(y, t)
        loss.backward()
        optimizer.update()
        lossData.append(loss.data)
    model.append(m)
    if keyNum == 1:
        label = 'where'
    elif keyNum == 2:
        label = 'angle'
    else:
        label = 'power'
    plt.plot(lossData, label=label)


def test(keyNum):
    inputData = np.empty((0, 11), dtype=np.float32)
    for i in range(trainSize):
        vec = str(df_train.iloc[i, 0])
        v = np.zeros(11, dtype=np.float32)
        for j in range(11):
            v[j] = float(vec[j+1])
        IN = np.array([v], dtype=np.float32)
        inputData = np.append(inputData, IN, axis=0)
    x = Variable(inputData)
    t = model[keyNum-1](x)

    for i in range(10):
        print(str(x[i])+"=", t.data[i])


batchsize = 100
n_epoch = 20

df_origin = pd.read_csv('./logs.csv', sep=',')
df_goodReward = df_origin[abs(df_origin['reward']) > 20]
df = df_origin.sample(frac=1)

dSize = len(df)
trainSize = int(dSize/5000)
testSize = dSize-trainSize


df_train = df[:trainSize]
df_test = df[trainSize+1: dSize]

model = []
for i in range(3):
    train(i+1)
keyName = ['where','angle','power']
for i in range(3):
    print(keyName[i])
    test(i+1)

plt.xlabel("No of trails")
plt.ylabel("loss")
plt.title("or.py")
plt.legend()
plt.show()
