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
    y = np.empty((0, 2), dtype=np.float32)
    for i in range(len(df)):
        vec = str(df.iloc[i, 0])
        v = np.zeros(89, dtype=np.float32)
        for j in range(89):
            v[j] = float(vec[j])
        IN = np.array([v], dtype=np.float32)
        o = np.zeros(2, dtype=np.float32)
        if float(df.iloc[i, 2]) < mean[count]:
            ans = 0
        else:
            ans = 1
        o[ans] = 1
        OUT = np.array([o], dtype=np.float32)
        x = np.append(x, IN, axis=0)
        y = np.append(y, OUT, axis=0)
    return x, y


def getDF(df_origin):
    df_train = df_origin
    return df_train


def train(x, y):
    # ネットワーク定義
    model = Sequential()
    model.add(Dense(256, input_dim=89, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    history = model.fit(x, y,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2)
    return model, history


def test(models, histories):
    for i in range(len(models)):
        model = models[i]
        history = histories[i]
        plt.plot(range(1, epochs+1),
                 history.history['loss'], label="loss"+str(i))
        plt.plot(range(1, epochs+1),
                 history.history['acc'], label="acc"+str(i))
        plt.xlabel('Epochs')
        plt.ylabel('')
        plt.legend()
        model.save("shotModel"+str(count)+".h5")
    plt.show()


def main():
    df_train = getDF(df)
    print("get train data")
    x_train, y_train = getData(df_train)
    print('strat train')
    model, history = train(x_train, y_train)
    return model, history


if __name__ == "__main__":
    data_cut = 1
    batch_size = 32
    epochs = 10
    models = []
    histories = []
    count = 0
    mean = [8.065887066940327, 7.318647432870903, 4.612291046358577, 5.531750416242592,
            4.921855637006497, 6.954194110411047, 8.473135741028479, 7.7882775466769445]
    for i in range(8):
        df = pd.read_csv('testLogs.csv', sep=',', header=None, names=(
            'vector', 'shot', 'reward'))
        df = df[df['shot'] == 0]
        model, history = main()
        models.append(model)
        histories.append(history)
        count += 1
    test(models, histories)
