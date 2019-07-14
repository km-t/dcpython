import numpy as np
from keras import metrics, callbacks
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import rmsprop
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# x_train：学習の入力(今回の場合はvec)
# y_train：出力(whereについては、[1,0,0,:0]  [0,1,0,:0]  [0,0,1,:0])の30通り
#   6p+3a+wで計算可能


def getInputData(df):
    x = np.empty((0, inputSize), dtype=np.float32)

    for i in tqdm(range(len(df))):
        vec = str(df.iloc[i, 0])
        v = np.zeros(inputSize, dtype=np.float32)
        for j in range(inputSize):
            v[j] = float(vec[j])
        IN = np.array([v], dtype=np.float32)
        x = np.append(x, IN, axis=0)
    return x


def getOutputData(df, key):
    if key == 0:  # w
        y = np.empty((0, 3), dtype=np.float32)
    if key == 1:  # a
        y = np.empty((0, 2), dtype=np.float32)
    if key == 2:  # p
        y = np.empty((0, 5), dtype=np.float32)
    if key == 3:  # s
        y = np.empty((0, 8), dtype=np.float32)

    for i in tqdm(range(len(df))):
        ans = int(df.iloc[i, key+1])
        if key == 0:
            o = np.zeros(3, dtype=np.float32)
            o[ans] = 1
        if key == 1:
            o = np.zeros(2, dtype=np.float32)
            o[ans] = 1
        if key == 2:
            o = np.zeros(5, dtype=np.float32)
            pData = [4, 6, 8, 10, 12]
            for i in range(len(pData)):
                if pData[i] == ans:
                    ans = i
                    break
            o[ans] = 1
        if key == 3:
            o = np.zeros(8, dtype=np.float32)
            ans = int(df.iloc[i, key+2])
            o[ans] = 1
        OUT = np.array([o], dtype=np.float32)
        y = np.append(y, OUT, axis=0)
    return y


def getDF(df):
    df = df.sample(frac=1)
    return df


def train(x, y, out_dim):
    # ネットワーク定義
    model = Sequential()
    model.add(Dense(128, input_dim=inputSize, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    for _ in range(3):
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(out_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='RMSprop',
                  metrics=[metrics.categorical_accuracy])
    #es = callbacks.EarlyStopping(monitor='loss', patience=0, verbose=0, mode='auto')
    history = model.fit(x, y,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2)
    return model, history


def plotScore(histories):
    fig, (axs, axw, axa, axp) = plt.subplots(
        ncols=3, figsize=(10, 4), sharex=True)

    axs.plot(histories[0].history['loss'], label="loss")
    axs.plot(histories[0].history['categorical_accuracy'], label="acc")
    axs.set_title('shot')
    axs.set_xlabel('epochs')
    axs.set_ylim(-0.5, 4)
    axs.grid(True)

    axw.plot(histories[1].history['loss'], label="loss")
    axw.plot(histories[1].history['categorical_accuracy'], label="acc")
    axw.set_title('power')
    axw.set_xlabel('epochs')
    axw.set_ylim(-0.5, 4)
    axw.grid(True)

    axa.plot(histories[2].history['loss'], label="loss")
    axa.plot(histories[2].history['categorical_accuracy'], label="acc")
    axa.set_title('where')
    axa.set_xlabel('epochs')
    axa.set_ylim(-0.5, 4)
    axa.grid(True)

    axp.plot(histories[3].history['loss'], label="loss")
    axp.plot(histories[3].history['categorical_accuracy'], label="acc")
    axp.set_title('angle')
    axp.set_xlabel('epochs')
    axp.set_ylim(-0.5, 4)
    axp.grid(True)

    fig.show()
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
    ans = np.argmax(y)
    print("input = {}\npredict = {}\nanswer = {}".format(d, y, ans))


def main():
    dfw_train = getDF(dfw)
    dfa_train = getDF(dfa)
    dfp_train = getDF(dfp)
    dfs_train = getDF(dfs)
    print("get input data")
    xw_train = getInputData(dfw_train)
    xa_train = getInputData(dfa_train)
    xp_train = getInputData(dfp_train)
    xs_train = getInputData(dfs_train)
    print("getOutputData")
    w_out = getOutputData(dfw_train, 0)
    a_out = getOutputData(dfa_train, 1)
    p_out = getOutputData(dfp_train, 2)
    s_out = getOutputData(dfs_train, 3)
    train_data = [xs_train, xw_train, xa_train, xp_train]
    y_train = [s_out, w_out, a_out, p_out]
    y_size = [8, 3, 2, 5]
    print('strat train')
    histories = []
    models = []
    for i in range(len(y_train)):
        model, history = train(train_data[i], y_train[i], y_size[i])
        histories.append(history)
        models.append(model)
    testData = ["10000000000000000000001000000100000000100000010000000001000000000000000000000000000000000",
                "00100000000000000000001000000001000000010000100000000001010000000000000000000000000000000",
                "00100000000000000100000000001000000000100000000000000000000000000000010011000000000000000",
                "01000000000000000000100000100000000001000010000010000000010000000000000010100000000000000",
                "01000000000000000000010000010000000001000010001000000001100000000000000011000000000000000",
                "00010000000000000000010000000000010000001001000000000000000000000000000000000000000000000",
                "00000000000010000000000101000000000000010000000100000000000000000000000000000000000000000",
                "00100000000000000000001001000000000000010010000100000000000000000000000000100000000000000"]
    label = ["where", "angle", "power"]
    i = 0
    for model in models:
        print("{}".format(label[i]))
        model.save("../"+label[i]+'Model.h5', include_optimizer=False)
        i += 1
        for ind in testData:
            test(model, ind)
    plotScore(histories)


if __name__ == "__main__":
    dfw = pd.read_csv('../logs/balancedWhereLogs.csv', sep=',', header=None, names=(
        'vector', 'where', 'angle', 'power', 'reward', 'shot'))
    dfa = pd.read_csv('../logs/balancedAngleLogs.csv', sep=',', header=None, names=(
        'vector', 'where', 'angle', 'power', 'reward', 'shot'))
    dfp = pd.read_csv('../logs/balancedPowerLogs.csv', sep=',', header=None, names=(
        'vector', 'where', 'angle', 'power', 'reward', 'shot'))
    dfs = pd.read_csv('../logs/balancedLogs.csv', sep=',', header=None, names=(
        'vector', 'where', 'angle', 'power', 'reward', 'shot'))

    dfw = dfw.drop_duplicates()
    dfa = dfa.drop_duplicates()
    dfp = dfp.drop_duplicates()
    dfs = dfs.drop_duplicates()
    inputSize = 89
    data_cut = 1
    batch_size = 32
    epochs = 200
    print("shot data:{}\nwhere data:{}\nangle data:{}\npower data:{}\n".format(
        len(dfs), len(dfw), len(dfa), len(dfp)))
    main()
