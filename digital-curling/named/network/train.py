import numpy as np
from keras import metrics, callbacks
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import rmsprop
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


class Train:
    x = None
    y = None
    df = None
    key = None
    out_dim = None

    def getInputData(self):
        inputSize = 91
        x = np.empty((0, inputSize), dtype=np.float32)
        for i in tqdm(range(len(self.df))):
            vec = str(self.df.iloc[i, 0])
            v = np.zeros(inputSize, dtype=np.float32)
            for j in range(inputSize):
                v[j] = float(vec[j])
            IN = np.array([v], dtype=np.float32)
            x = np.append(x, IN, axis=0)
        return x

    def getOutputData(self):
        if self.key == 0:  # w
            y = np.empty((0, 3), dtype=np.float32)
            self.out_dim = 3
        if self.key == 1:  # a
            y = np.empty((0, 2), dtype=np.float32)
            self.out_dim = 2
        if self.key == 2:  # p
            y = np.empty((0, 5), dtype=np.float32)
            self.out_dim = 5
        if self.key == 3:  # s
            y = np.empty((0, 8), dtype=np.float32)
            self.out_dim = 8

        for i in tqdm(range(len(self.df))):
            ans = int(self.df.iloc[i, self.key+1])
            if self.key == 0:
                o = np.zeros(3, dtype=np.float32)
                o[ans] = 1
            if self.key == 1:
                o = np.zeros(2, dtype=np.float32)
                o[ans] = 1
            if self.key == 2:
                o = np.zeros(5, dtype=np.float32)
                pData = [4, 6, 8, 10, 12]
                for i in range(len(pData)):
                    if pData[i] == ans:
                        ans = i
                        break
                o[ans] = 1
            if self.key == 3:
                o = np.zeros(8, dtype=np.float32)
                ans = int(self.df.iloc[i, self.key+2])
                o[ans] = 1
            OUT = np.array([o], dtype=np.float32)
            y = np.append(y, OUT, axis=0)
        return y

    def train(self):
        # ネットワーク定義
        model = Sequential()
        model.add(Dense(128, input_dim=91, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        for _ in range(3):
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(self.out_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                           optimizer='RMSProp',
                           metrics=[metrics.categorical_accuracy])
        history = model.fit(self.x, self.y,
                            epochs=200,
                            batch_size=256,
                            validation_split=0.3)
        return model, history

    def __init__(self, _df, _key):
        self.df = _df
        self.key = _key
        self.x = self.getInputData()
        self.y = self.getOutputData()


def plotScore(histories):
    fig, (axw, axa, axp, axs) = plt.subplots(
        ncols=4, figsize=(10, 4), sharex=True)

    axs.plot(histories[3].history['loss'], label="loss")
    axs.plot(histories[3].history['categorical_accuracy'], label="acc")
    axs.set_title('shot')
    axs.set_xlabel('epochs')
    axs.set_ylim(-0.5, 4)
    axs.grid(True)

    axw.plot(histories[0].history['loss'], label="loss")
    axw.plot(histories[0].history['categorical_accuracy'], label="acc")
    axw.set_title('power')
    axw.set_xlabel('epochs')
    axw.set_ylim(-0.5, 4)
    axw.grid(True)

    axa.plot(histories[1].history['loss'], label="loss")
    axa.plot(histories[1].history['categorical_accuracy'], label="acc")
    axa.set_title('where')
    axa.set_xlabel('epochs')
    axa.set_ylim(-0.5, 4)
    axa.grid(True)

    axp.plot(histories[2].history['loss'], label="loss")
    axp.plot(histories[2].history['categorical_accuracy'], label="acc")
    axp.set_title('angle')
    axp.set_xlabel('epochs')
    axp.set_ylim(-0.5, 4)
    axp.grid(True)

    fig.show()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    models = []
    histories = []
    for i in range(4):
        file = '../logs/log'+str(i)+'.csv'
        print('train by {}'.format(file))
        df = pd.read_csv(file, sep=',', header=None, names=(
            'vector', 'where', 'angle', 'power', 'reward', 'shot'))
        df = df.drop_duplicates()
        df = df.sample(frac=1)
        trainer = Train(df, i)
        model, history = trainer.train()
        models.append(model)
        histories.append(history)
        model.save('model'+str(i)+'.h5')
    plotScore(histories)
