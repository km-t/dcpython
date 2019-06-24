import numpy as np
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import rmsprop
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# x_train：学習の入力(今回の場合はvec)
# y_train：出力(whereについては、[1,0,0,:0]  [0,1,0,:0]  [0,0,1,:0])の30通り
#   6p+3a+wで計算可能


def getData(df):
    x = np.empty((0, inputSize), dtype=np.float32)
    y = np.empty((0, 1), dtype=np.float32)
    for i in tqdm(range(len(df))):
        vec = str(df.iloc[i, 0])
        v = np.zeros(inputSize, dtype=np.float32)
        for j in range(89):
            v[j] = float(vec[j])
        inAdd = np.zeros(30, dtype=np.float32)
        where = int(df.iloc[i, 1])
        angle = int(df.iloc[i, 2])
        p = int(df.iloc[i, 3])
        pData = [3, 5, 7, 12, 16]
        power = 0
        for k in range(len(pData)):
            if pData[k] == p:
                power = k
        ans = power*6+angle*3+where
        inAdd[ans] = 1
        for j in range(30):
            v[89+j] = inAdd[j]
        IN = np.array([v], dtype=np.float32)
        o = np.zeros(1, dtype=np.float32)
        o[0] = float(df.iloc[i, 4])
        OUT = np.array([o], dtype=np.float32)
        x = np.append(x, IN, axis=0)
        y = np.append(y, OUT, axis=0)
    return x, y


def getDF(df_origin):
    df_origin = df_origin.sample(frac=1)
    df_train = df_origin[:trainNum].sample(frac=1)
    return df_train


def train(x, y):
    # ネットワーク定義
    model = Sequential()
    # activations = ['relu', 'elu', 'selu', 'softplus', 'softsign', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear', 'softmax']
    model.add(Dense(256, input_dim=inputSize, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    for _ in range(3):
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_absolute_error',
                  optimizer='Adam',
                  metrics=['accuracy'])
    history = model.fit(x, y,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.3)

    return model, history


def convert30to3(x):
    #x = power*6+angle*3+where
    where = 0
    angle = 0
    power = 0
    power = (int)(x/6)
    x = x-6*power
    angle = (int)(x/3)
    x = x-3*power
    where = x
    return [where, angle, power]


def plotScore(history):
    plt.plot(history.history['loss'], label="loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.plot(history.history['acc'], label="acc")
    plt.plot(history.history['val_acc'], label="val_acc")
    plt.xlabel('Epochs')
    plt.ylabel('')
    plt.legend()
    plt.show()


def main():
    df_train = getDF(df)
    print("get train data")
    x_train, y_train = getData(df_train)
    print('strat train')
    model, history = train(x_train, y_train)
    plotScore(history)


if __name__ == "__main__":
    df = pd.read_csv('../logs/highScoreLogsVer2.csv', sep=',', header=None, names=(
        'vector', 'where', 'angle', 'power', 'reward'))
    df = df.sample(frac=1)
    df = df.drop_duplicates()
    dataSize = len(df)
    inputSize = 119
    data_cut = 1
    batch_size = 32
    trainNum = int(dataSize*data_cut)
    epochs = (int)(trainNum/batch_size)
    epochs = 20
    print("all data:{}\ntrain data:{}".format(dataSize, trainNum))
    main()
