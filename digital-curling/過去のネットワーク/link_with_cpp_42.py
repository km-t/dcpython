import keras
from flask import Flask, render_template, request, redirect, url_for
import math
import numpy as np
import tensorflow as tf
from keras import backend as K

app = Flask(__name__)


def getDist(stone):
    ans = math.sqrt((stone[0]-2.375)**2 + (stone[1]-4.88)**2)
    return ans


def getDegree(x, y):
    radian = math.atan2(x-2.375, y-4.88)
    return 180*radian/math.pi


def getRank(board, target):
    stones = []
    for i in range(16):
        stones.append([board[i*2], board[i*2+1]])
    dists = []
    for i in range(16):
        if stones[i][0]+stones[i][1] == 0.00:
            dists.append(99999)
        else:
            dists.append(getDist(stones[i]))
    sort = []
    for i in range(16):
        sort.append(dists[i])
    for j in range(16):
        for i in range(15):
            if sort[i] > sort[i+1]:
                tmp = sort[i]
                sort[i] = sort[i+1]
                sort[i+1] = tmp
    for i in range(16):
        if dists[target] == sort[i]:
            return i


def getVector(board, target, isMine):
    """
    0~15:rank
    16~23:x
    24~35:y
    36~41:dist
    42:isMine
    43~55:angle
    """
    ans = ""
    rank = getRank(board, target)
    x = board[target*2]
    y = board[target*2+1]
    if x+y == 0:
        return "1111111111111111111111111111111111111111111111111111111"

    degree = getDegree(x, y)
    dist = getDist([x, y])
    if rank == 0:
        ans += "1000000000000000"
    elif rank == 1:
        ans += "0100000000000000"
    elif rank == 2:
        ans += "0010000000000000"
    elif rank == 3:
        ans += "0001000000000000"
    elif rank == 4:
        ans += "0000100000000000"
    elif rank == 5:
        ans += "0000010000000000"
    elif rank == 6:
        ans += "0000001000000000"
    elif rank == 7:
        ans += "0000000100000000"
    elif rank == 8:
        ans += "0000000010000000"
    elif rank == 9:
        ans += "0000000001000000"
    elif rank == 10:
        ans += "0000000000100000"
    elif rank == 11:
        ans += "0000000000010000"
    elif rank == 12:
        ans += "0000000000001000"
    elif rank == 13:
        ans += "0000000000000100"
    elif rank == 14:
        ans += "0000000000000010"
    else:
        ans += "0000000000000001"

    if x < 2.375-1.83:
        ans += "10000000"
    elif 2.375-1.83 <= x < 2.375-1.22:
        ans += "01000000"
    elif 2.375-1.22 <= x < 2.375-0.61:
        ans += "00100000"
    elif 2.375-0.61 <= x < 2.375:
        ans += "00010000"
    elif 2.375 <= x < 2.375+0.61:
        ans += "00001000"
    elif 2.375+0.61 <= x < 2.375+1.22:
        ans += "00000100"
    elif 2.375+1.22 <= x < 2.375+1.83:
        ans += "00000010"
    elif 2.375+1.83 <= x:
        ans += "00000001"
    else:
        ans += "00000000"

    if y < 4.88-1.83:
        ans += "100000000000"
    elif 4.88-1.83 <= y < 4.88-1.22:
        ans += "010000000000"
    elif 4.88-1.22 <= y < 4.88-0.61:
        ans += "001000000000"
    elif 4.88-0.61 <= y < 4.88:
        ans += "000100000000"
    elif 4.88 <= y < 4.88+0.61:
        ans += "000010000000"
    elif 4.88+0.61 <= y < 4.88+1.22:
        ans += "000001000000"
    elif 4.88+1.22 <= y < 4.88+1.83:
        ans += "000000100000"
    elif 4.88+1.83 <= y < 4.88+2.68:
        ans += "000000010000"
    elif 4.88+2.68 <= y < 4.88+3.53:
        ans += "000000001000"
    elif 4.88+3.53 <= y < 4.88+4.38:
        ans += "000000000100"
    elif 4.88+4.38 <= y < 4.88+5.23:
        ans += "000000000010"
    elif 4.88+5.23 <= y:
        ans += "000000000001"
    else:
        ans += "000000000000"

    if dist < 0.61:
        ans += "100000"
    elif 0.61 <= dist < 1.22:
        ans += "010000"
    elif 1.22 <= dist < 1.83:
        ans += "001000"
    elif 1.83 <= dist < 3.05:
        ans += "000100"
    elif 3.05 <= dist < 4.27:
        ans += "000010"
    elif 4.27 <= dist < 5.49:
        ans += "000001"
    else:
        ans += "000000"

    if target % 2 == isMine:
        ans += "1"
    else:
        ans += "0"
    if 0 <= degree < 30:
        ans += "100000000000"
    elif 30 <= degree < 60:
        ans += "010000000000"
    elif 60 <= degree < 90:
        ans += "001000000000"
    elif 90 <= degree < 120:
        ans += "000100000000"
    elif 120 <= degree < 150:
        ans += "000010000000"
    elif 150 <= degree < 180:
        ans += "000001000000"
    elif 180 <= degree < 210:
        ans += "000000100000"
    elif 210 <= degree < 240:
        ans += "000000010000"
    elif 240 <= degree < 270:
        ans += "000000001000"
    elif 270 <= degree < 300:
        ans += "000000000100"
    elif 300 <= degree < 330:
        ans += "000000000010"
    elif 330 <= degree < 360:
        ans += "000000000001"
    else:
        ans += "000000000000"
    return ans


def convertToFloat(Board):
    Board = Board.split(",")
    board = np.zeros(32, dtype=float)
    for i in range(32):
        board[i] = float(Board[i])
    return board


def load_model():
    global model, graph
    model = keras.models.load_model('../model.h5', compile=False)
    graph = tf.get_default_graph()


def adjust(pre):
    def round(x): return (x*2+1)//2
    pre[0][0] = round(pre[0][0])
    pre[0][1] = round(pre[0][1])
    return pre


def convertAns(pre):
    ans = ""
    ans = str(pre[0][0])+","+str(pre[0][1])+","+str(pre[0][2])+","
    return ans


@app.route('/<Board>', methods=['GET', 'POST'])
def hello(Board):
    answer = ""
    count = 0
    inputSize = 42
    if Board != 'favicon.ico':
        board = convertToFloat(Board)
        wantNo = []
        isExist = False
        for i in range(16):
            if board[i*2]+board[i*2+1] != 0:
                wantNo.append(i)
                isExist = True
        count = len(wantNo)
        answer = str(count)+","
        if isExist:
            for i in wantNo:
                vecs = getVector(board, i, i % 2)
                inputData = np.zeros((0, inputSize), dtype=np.float32)
                v = np.zeros(inputSize, dtype=np.float32)
                for j in range(len(vecs)):
                    v[j] = float(vecs[j])
                inputData = np.array([v], dtype=np.float32)
                with graph.as_default():
                    pre = model.predict(inputData)
                    pre = adjust(pre)
                    ans = convertAns(pre)
                    ans += str(i)+","
                    answer += ans
            answer = answer[:-1]
        else:
            answer += "-1,-1,-1,-1"
    return answer


if __name__ == '__main__':
    load_model()
    app.run(debug=False, port=80)
