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


def getData(df):
    x = np.empty((0, 89), dtype=np.float32)
    y = np.empty((0, 3), dtype=np.float32)
    for i in tqdm(range(len(df))):
        vec = str(df.iloc[i, 0])
        v = np.zeros(89, dtype=np.float32)
        for j in range(89):
            v[j] = float(vec[j])
        IN = np.array([v], dtype=np.float32)
        o = np.zeros(3, dtype=np.float32)
        o[0] = int(df.iloc[i, 1])
        o[1] = int(df.iloc[i, 2])
        o[2] = int(df.iloc[i, 3])
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
    model.add(Dense(256, input_dim=89, activation=activation))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation=activation))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation=activation))
    model.compile(loss='mean_squared_error',
                  optimizer='Adam',
                  metrics=['accuracy'])
    history = model.fit(x, y,
                        epochs=epochs,
                        batch_size=batch_size)
    return model, history


def test(model, history, x, y):
    plt.plot(history.history['loss'], label="loss")
    plt.plot(history.history['acc'], label="acc")
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
    trainSize = 0.8
    data_cut = 0.5
    batch_size = 128
    epochs = 50
    df = pd.read_csv('../logs/logsVer2.csv', sep=',', header=None, names=(
        'vector', 'where', 'angle', 'power', 'reward'))
    print(int(len(df)*data_cut))
    main()
