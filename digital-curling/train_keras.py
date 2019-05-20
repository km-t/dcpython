import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import rmsprop
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

#x_train：学習の入力(今回の場合はvec)
#y_train：出力(whereについては、[1,0,0,:0]  [0,1,0,:0]  [0,0,1,:0])の30通り
#   6p+3a+wで計算可能


def culcAccuracy(model, testx, y):
    pre = np.empty((0,30), dtype = np.float32)
    for x in tqdm(testx):
        x = np.array([x], dtype=np.float32)
        arg = np.argmax(model.predict(x))
        p = np.zeros(30, dtype=np.float32)
        p[arg] = 1
        pre = np.append(pre, [p], axis=0)
    ans = y
    count = 0
    true = 0
    for i in range(len(pre)):    
        isEqual = True
        for j in range(len(pre[0])):
            if pre[i][j]==ans[i][j]:
                pass
            else:
                isEqual = False
        if isEqual:
            true += 1
        count += 1
    print(count, true, true/count)


def getData(df):
    x = np.empty((0, 43), dtype=np.float32)
    y = np.empty((0,30), dtype=np.float32)
    for i in range(len(df)):
        vec = str(df.iloc[i, 0])
        v = np.zeros(43, dtype=np.float32)
        for j in range(43):
            v[j] = float(vec[j])
        IN = np.array([v], dtype=np.float32)
        o = np.zeros(30, dtype=np.float32)
        where = int(df.iloc[i,1])
        angle = int(df.iloc[i,2])
        p = int(df.iloc[i,3])
        pData = [3,5,7,12,16]
        power = 0
        for k in range(len(pData)):
            if pData[k]==p:
                power = k
        ans = power*6+angle*3+where
        #o[ans] = int(df.iloc[i,4])
        o[ans] = 1
        OUT = np.array([o], dtype=np.float32)
        x = np.append(x, IN, axis=0)
        y = np.append(y, OUT, axis=0)
    return x,y

def getDF(df_origin):
    trainNum = int(len(df_origin)*DataSize)
    testNum = int((len(df_origin)-trainNum)*DataSize)
    df_origin=df_origin.sample(frac=1)
    df_train = df_origin[:trainNum]
    df_test = df_origin[trainNum+1: trainNum+testNum]
    return df_train, df_test


def train(x, y):
    # ネットワーク定義
    model = Sequential()
    activations = ['relu','elu','selu','softplus','softsign','tanh','sigmoid','hard_sigmoid','linear','softmax']
    activation = activations[6]#0~8
    model.add(Dense(256, input_dim=43, activation=activation))
    model.add(Dense(128, activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dense(128, activation=activation))
    model.add(Dense(30, activation=activation))
    model.compile(loss='categorical_crossentropy',
                optimizer='Adam',
                metrics=['accuracy'])
    history = model.fit(x, y,
            epochs=epochs,
            batch_size=batch_size)
    return model, history

def test(model, history, x, y):
    culcAccuracy(model, x, y)
    plt.plot(range(1, epochs+1), history.history['loss'], label="loss")
    plt.plot(range(1, epochs+1), history.history['acc'], label="acc")
    plt.xlabel('Epochs')
    plt.ylabel('')
    plt.legend()
    #plt.show()


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
    DataSize = 0.1
    batch_size=32
    epochs=50
    df = pd.read_csv('logs.csv',sep=',',header=None,names=('vector','where','angle','power','reward'))
    df = df[df['reward']>=2]
    print(df)
    print(len(df))
    main()


