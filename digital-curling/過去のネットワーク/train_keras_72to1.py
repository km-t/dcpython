import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import rmsprop
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# x_train：学習の入力(今回の場合はvec)
# y_train：出力(whereについては、[1,0,0,:0]  [0,1,0,:0]  [0,0,1,:0])の30通り
#   6p+3a+wで計算可能


def culcAccuracy(model, testx, y):
    pre = np.empty((0, 1), dtype=np.float32)
    for x in tqdm(testx):
        x = np.array([x], dtype=np.float32)
        print(x)
        pp = round(float(model.predict(x)))
        p = np.zeros(1, dtype=np.float32)
        p[0] = pp
        pre = np.append(pre, [p], axis=0)
    ans = y
    count = 0
    true = 0
    squaredCount = 0
    acc = []
    for i in range(len(pre)):
        squaredCount += 1 - ((pre[i] - ans[i])**2)/4
        if pre[i] == ans[i]:
            true += 1
        count += 1
        if i % int(len(pre)/epochs) == 0:
            acc.append(true/count)
    print(count, true, true/count)
    print(squaredCount/count)
    print(len(acc))
    print(len(pre), int(len(pre)/epochs))
    return acc


def getData(df):
    x = np.empty((0, 72), dtype=np.float32)
    y = np.empty((0, 1), dtype=np.float32)
    for i in range(len(df)):
        vec = str(df.iloc[i, 0])
        v = np.zeros(72, dtype=np.float32)
        for j in range(42):
            v[j] = float(vec[j])
        where = int(df.iloc[i, 1])
        angle = int(df.iloc[i, 2])
        p = int(df.iloc[i, 3])
        pData = [3, 5, 7, 12, 16]
        power = 0
        for k in range(len(pData)):
            if pData[k] == p:
                power = k
        ans = power*6+angle*3+where
        for j in range(30):
            if j == ans:
                v[j] = 1
            else:
                v[j] = 0
        IN = np.array([v], dtype=np.float32)
        o = np.zeros(1, dtype=np.float32)
        o[0] = int(df.iloc[i, 4])
        OUT = np.array([o], dtype=np.float32)
        x = np.append(x, IN, axis=0)
        y = np.append(y, OUT, axis=0)
    return x, y


def getDF(df_origin):
    trainNum = int(len(df_origin)*trainSize)
    testNum = int(len(df_origin)-trainNum)
    trainNum = int(trainNum*data_cut)
    testNum = int(testNum*data_cut)
    df_origin = df_origin.sample(frac=1)
    df_train = df_origin[:trainNum].sample(frac=1)
    df_test = df_origin[trainNum+1: trainNum+testNum].sample(frac=1)
    return df_train, df_test


def train(x, y):
    # ネットワーク定義
    model = Sequential()
    activations = ['relu', 'elu', 'selu', 'softplus', 'softsign',
                   'tanh', 'sigmoid', 'hard_sigmoid', 'linear', 'softmax']
    activation = activations[0]  # 0~9
    model.add(Dense(256, input_dim=72, activation=activation))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation=activation))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation=activation))
    model.compile(loss='mean_squared_error',
                  optimizer='Adam',
                  metrics=['accuracy'])
    history = model.fit(x, y,
                        epochs=epochs,
                        batch_size=batch_size)
    return model, history


def test(model, history, x, y):
    myAcc = culcAccuracy(model, x, y)
    plt.plot(history.history['loss'], label="loss")
    plt.plot(history.history['acc'], label="acc")
    plt.plot(myAcc, label='myAcc')
    plt.xlabel('Epochs')
    plt.ylabel('')
    plt.legend()
    plt.show()


def main():
    df_train, df_test = getDF(df)
    print("get train data")
    x_train, y_train = getData(df_train)
    print("get test data")
    x_test, y_test = getData(df_test)
    print('strat train')
    model, history = train(x_train, y_train)

    test(model, history, x_test, y_test)


if __name__ == "__main__":
    trainSize = 0.7
    data_cut = 0.0001
    batch_size = 2048
    epochs = 500
    df = pd.read_csv('logs.csv', sep=',', header=None, names=(
        'vector', 'where', 'angle', 'power', 'reward'))
    # df = df[(df['vector']=='1000010000000000001000000010000000000000000')]
    # df = df[df['reward']>=2]
    print(len(df))
    main()
