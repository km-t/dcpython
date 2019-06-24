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
    y = np.empty((0, 3), dtype=np.float32)
    for i in tqdm(range(len(df))):
        vec = str(df.iloc[i, 0])
        v = np.zeros(inputSize, dtype=np.float32)
        for j in range(inputSize):
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
    df_origin = df_origin.sample(frac=1)
    df_train = df_origin[:trainNum].sample(frac=1)
    return df_train


def train(x, y, loss, opt):
    # ネットワーク定義
    model = Sequential()
    model.add(Dense(128, input_dim=inputSize, activation='relu'))
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
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='relu'))
    model.compile(loss=loss,
                  optimizer=opt)
    history = model.fit(x, y,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0,
                        validation_split=0.3)
    return history, model


def plotScore(histories):
    """
    plt.plot(history.history['loss'], label="loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.plot(history.history['acc'], label="acc")
    plt.plot(history.history['val_acc'], label="val_acc")
    """
    i = 0
    for history in histories:
        plt.plot(history.history['loss'], label="acc No.{}".format(i))
        i += 1
    plt.xlabel('Epochs')
    plt.ylabel('')
    plt.legend()
    plt.show()


def test(model, inp):
    d = inp
    inD = np.zeros(89, dtype=np.float32)
    for i in range(89):
        inD[i] = float(d[i])
    inputData = np.empty((0, 89), dtype=np.float32)
    IN = np.array([inD], dtype=np.float32)
    inputData = np.append(inputData, IN, axis=0)
    y = model.predict(inputData)
    print("input = {}\npredict = {}".format(d, y))


def main():
    df_train = getDF(df)
    print("get train data")
    x_train, y_train = getData(df_train)
    print('strat train')
    optimizers = ['SGD', 'Adam', 'RMSprop', 'Adagrad', 'Adamax', 'Nadam']
    losses = ['mean_absolute_error', "mean_squared_error",
              "hinge", 'cosine_proximity']

    optimizers = ['Nadam']
    losses = ['mean_absolute_error', "mean_squared_error",
              "hinge"]

    history = []
    i = 0

    testData = ["10000000000000000000001000000100000000100000010000000001000000000000000000000000000000000",
                "00100000000000000000001000000001000000010000100000000001010000000000000000000000000000000",
                "00100000000000000100000000001000000000100000000000000000000000000000010011000000000000000",
                "01000000000000000000100000100000000001000010000010000000010000000000000010100000000000000",
                "01000000000000000000010000010000000001000010001000000001100000000000000011000000000000000",
                "00010000000000000000010000000000010000001001000000000000000000000000000000000000000000000"]

    for loss in losses:
        for opt in optimizers:
            print("{}steps\nloss_function = {}\noptimizer = {}".format(i, loss, opt))
            i += 1
            his, model = train(x_train, y_train, loss, opt)
            history.append(his)
            for d in testData:
                test(model, d)

    plotScore(history)
    # model.save('model.h5', include_optimizer=False)


if __name__ == "__main__":
    df = pd.read_csv('../logs/highScoreLogsVer2.csv', sep=',', header=None, names=(
        'vector', 'where', 'angle', 'power', 'reward'))
    df = df.drop_duplicates()
    inputSize = 89
    dataSize = len(df)
    data_cut = 0.1
    batch_size = 128
    trainNum = int(dataSize*data_cut)
    epochs = (int)(trainNum/batch_size)
    epochs = 50
    print("all data:{}\ntrain data:{}".format(dataSize, trainNum))
    main()


"""
def getData(df):
    x = np.empty((0, inputSize), dtype=np.float32)
    y = np.empty((0, 3), dtype=np.float32)
    for i in tqdm(range(len(df))):
        vec = str(df.iloc[i, 0])
        v = np.zeros(inputSize, dtype=np.float32)
        for j in range(inputSize):
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
    df_origin = df_origin.sample(frac=1)
    df_train = df_origin[:trainNum].sample(frac=1)
    df_test = df_origin[trainNum+1: trainNum+testNum].sample(frac=1)
    return df_train, df_test


def train(x, y):
    # ネットワーク定義
    model = Sequential()
    model.add(Dense(128, input_dim=inputSize, activation='relu'))
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
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='relu'))
    model.compile(loss='mean_absolute_error',
                  optimizer='Adam',
                  metrics=['mae','acc'])
    history = model.fit(x, y,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1)
    return model, history


def test(model, history, x, y):
    score = model.evaluate(x, y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    plt.plot(history.history['loss'], label="loss")
    plt.plot(history.history['acc'], label="acc")
    plt.xlabel('Epochs')
    plt.ylabel('')
    plt.ylim(-1, 2)
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
    model.save('model.h5', include_optimizer=False)

if __name__ == "__main__":
    df = pd.read_csv('../logs/highScoreLogsVer2.csv', sep=',', header=None, names=(
        'vector', 'where', 'angle', 'power', 'reward'))
    df = df.drop_duplicates()
    inputSize = 89
    dataSize = len(df)
    trainSize = 0.7
    data_cut = 1
    batch_size = 32
    epochs = (int)((dataSize*trainSize*data_cut)/batch_size)
    epochs = 50
    trainNum = int(dataSize*trainSize)
    testNum = int(data_cut*(dataSize-trainNum))
    trainNum = int(trainNum*data_cut)
    print("all data:{}\ntrain data:{}\ntest data:{}".format(dataSize,trainNum,testNum))
    main()
"""
