from tqdm import tqdm
import numpy as np
import pandas as pd
import math
import tkinter as tk
import random
stoneR = 0.145


def isMyGuard(board, target, isMine):
    for i in range(16):
        if i != target:
            if board[target*2] >= board[i*2] and board[target*2] + float(-1) <= board[i*2]:
                if board[i*2+1] >= board[target*2+1] and board[i*2+1] <= board[target*2+1] + stoneR*6:
                    if i % 2 == isMine:
                        return True
                    else:
                        return False
            if board[target*2] >= board[i*2] and board[target*2] + float(1) <= board[i*2]:
                if board[i*2+1] >= board[target*2+1] and board[i*2+1] <= board[target*2+1] + stoneR*6:
                    if i % 2 == isMine:
                        return True
                    else:
                        return False
    return False


def getDistBet(x, y, x2, y2):
    return math.sqrt((x-x2)**2 + (y-y2)**2)


def isGuarded(board, target):
    for i in range(16):
        if i != target:
            if board[target*2] >= board[i*2] and board[target*2] + float(-1) <= board[i*2]:
                if board[i*2+1] >= board[target*2+1] and board[i*2+1] <= board[target*2+1] + stoneR*6:
                    return True
            if board[target*2] >= board[i*2] and board[target*2] + float(1) <= board[i*2]:
                if board[i*2+1] >= board[target*2+1] and board[i*2+1] <= board[target*2+1] + stoneR*6:
                    return True
    return False


def canGuard(board, target):
    count = 0
    for i in range(16):
        if i != target:
            if board[target*2] >= board[i*2] and board[target*2] + float(-1) <= board[i*2]:
                if board[i*2+1] <= board[target*2+1]:
                    count += 1
            if board[target*2] >= board[i*2] and board[target*2] + float(1) <= board[i*2]:
                if board[i*2+1] <= board[target*2+1]:
                    count += 1
    return count


def isMyFreeze(board, target, isMine):
    for i in range(16):
        if i != target:
            if getDistBet(board[target*2], board[target*2+1], board[i*2], board[i*2+1]) < stoneR*1.5:
                if board[target*2+1] < board[i*2+1]:
                    if isMine == i % 2:
                        return True
                    else:
                        return False
    return False


def isFreezed(board, target):
    for i in range(16):
        if i != target:
            if getDistBet(board[target*2], board[target*2+1], board[i*2], board[i*2+1]) < stoneR*1.5:
                if board[target*2+1] < board[i*2+1]:
                    return True
    return False


def canFreezed(board, target, count):
    if count > 15:
        return count
    for i in range(16):
        if i != target:
            if getDistBet(board[target*2], board[target*2+1], board[i*2], board[i*2+1]) < stoneR*3:
                if board[target*2+1] > board[i*2+1]:
                    count += 1
                    canFreezed(board, i, count)
    return count


def outStone(stone):
    root = tk.Tk()
    width = int(4.75*50)
    height = int(11.28*50)
    x = 2.375
    y = 4.88
    print(width, height)
    root.geometry("237x564")
    canvas = tk.Canvas(root, width=width, height=height)
    canvas.create_oval((2.375-1.83)*50, (4.88-1.83)*50,
                       (2.375+1.83)*50, (4.88+1.83)*50, fill='blue')
    canvas.create_oval((2.375-1.22)*50, (4.88-1.22)*50,
                       (2.375+1.22)*50, (4.88+1.22)*50, fill='white')
    canvas.create_oval((2.375-0.61)*50, (4.88-0.61)*50,
                       (2.375+0.61)*50, (4.88+0.61)*50, fill='red')
    canvas.create_line(2.375*50, 1.22*50, 2.375*50, 11.28*50)
    canvas.create_line(0, 4.88*50, 4.75*50, 4.88*50)
    canvas.create_oval((stone[0]-0.145)*50, (stone[1]-0.145)*50,
                       (stone[0]+0.145)*50, (stone[1]+0.145)*50, fill="yellow")

    canvas.place(x=0, y=0)
    root.mainloop()


def outBoard(board):
    root = tk.Tk()
    width = int(4.75*50)
    height = int(11.28*50)
    x = 2.375
    y = 4.88
    root.geometry("237x564")
    canvas = tk.Canvas(root, width=width, height=height)
    canvas.create_oval((2.375-1.83)*50, (4.88-1.83)*50,
                       (2.375+1.83)*50, (4.88+1.83)*50, fill='blue')
    canvas.create_oval((2.375-1.22)*50, (4.88-1.22)*50,
                       (2.375+1.22)*50, (4.88+1.22)*50, fill='white')
    canvas.create_oval((2.375-0.61)*50, (4.88-0.61)*50,
                       (2.375+0.61)*50, (4.88+0.61)*50, fill='red')
    canvas.create_line(2.375*50, 1.22*50, 2.375*50, 11.28*50)
    canvas.create_line(0, 4.88*50, 4.75*50, 4.88*50)
    for i in range(16):
        stone = [board[i*2], board[i*2+1]]
        if stone[0]+stone[1] != 0:
            if i % 2 == 0:
                canvas.create_oval((stone[0]-0.145)*50, (stone[1]-0.145)*50,
                                   (stone[0]+0.145)*50, (stone[1]+0.145)*50, fill="yellow")
            else:
                canvas.create_oval((stone[0]-0.145)*50, (stone[1]-0.145)*50,
                                   (stone[0]+0.145)*50, (stone[1]+0.145)*50, fill="black")

    canvas.place(x=0, y=0)
    root.mainloop()


def getDist(stone):
    ans = math.sqrt((stone[0]-2.375)**2 + (stone[1]-4.88)**2)
    if stone[0]+stone[1] == 0:
        return 999999999
    else:
        return ans


def getDegree(x, y):
    radian = math.atan2(x-2.375, y-4.88)
    return 180*radian/math.pi


def getVector(board, target, isMine):
    """
    0~15:rank 16
    16~23:x 8
    24~35:y 12
    36~41:dist 6
    42:isMine 1
    43~54:degree 12
    55:isGuarded 1
    56:isMyGuard 1
    57~71:canGuard 15
    72:isFreezed 1
    73:isMyFreeze 1
    74~88:canFreezed 15
    """
    ans = ""
    rank = getRank(board, target)
    x = board[target*2]
    y = board[target*2+1]
    if x+y == 0:
        return "11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111"

    degree = getDegree(x, y)
    dist = getDist([x, y])
    r = "0000000000000000"
    ans += r[:rank]+"1"+r[rank+1:]

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
    if isGuarded(board, target):
        ans += "1"
    else:
        ans += "0"
    if isMyGuard(board, target, isMine):
        ans += "1"
    else:
        ans += "0"

    guardNum = canGuard(board, target)
    r = "000000000000000"
    if guardNum < 15:
        ans += r[:guardNum]+"1"+r[guardNum+1:]
    # 73
    if isFreezed(board, target):
        ans += "1"
    else:
        ans += "0"
    if isMyFreeze(board, target, isMine):
        ans += "1"
    else:
        ans += "0"
    freezeNum = canFreezed(board, target, 0)
    r = "000000000000000"
    if freezeNum < 15:
        ans += r[:freezeNum]+"1"+r[freezeNum+1:]
    return ans


def getRank(board, target):
    stones = []
    for i in range(16):
        stones.append([board[i*2], board[i*2+1]])
    dists = []
    for i in range(16):
        if stones[i][0]+stones[i][1] <= 0.00:
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


def getBoardScore(board, turn):
    # turn==0: 0,2,4,6,8,10,12,14 is my stone
    ranks = []
    for i in range(16):
        ranks.append(getRank(board, i))
        # ranks[0]=0番目の石のランク
    no0Stone = 0
    for i in range(16):
        if ranks[i] == 0:
            no0Stone = i
            break
    scoredPlayer = no0Stone % 2
    score = 0
    for j in range(8):
        isGet = False
        for i in range(8):
            if board[(2*i+scoredPlayer)*2]+board[(2*i+scoredPlayer)*2+1] == 0:
                break
            if ranks[2*i+scoredPlayer] == j:
                score += 1
                isGet = True
        if not(isGet):
            break
    if turn != scoredPlayer:
        score *= -1
    return score


def getVectorScore(vec,  turn):
    weight = [
        [4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 1, 0, 0, 0, 0, 0, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5, -3, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        [6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 3, 2, 1, 0, 0, 0, 2,
            2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    ]
    if turn <= 9:
        state = 0
    else:
        state = 1
    score = 0
    for i in range(89):
        score += weight[state][i]*int(vec[i])
    return score


def getScore(board, turn):
    """
    if target % 2 == isMine:
        ans += "1"
    else:
        ans += "0"
    if turn%2==0:
        自分は先手
    """
    vecs = []
    for i in range(16):
        if board[i*2]+board[i*2+1] != 0:
            vec = getVector(board, i, turn % 2)
            vecs.append(vec)
    score = 0
    for i in range(len(vecs)):
        if vecs[i][42] == "1":
            isMine = 1
        else:
            isMine = -1
        score += getVectorScore(vecs[i], turn)*isMine
    return score


def getStoneNum(board):
    count = 0
    for i in range(16):
        if board[2*i]+board[2*i+1] != 0:
            count += 1
    return count


def main():
    file = "C:/Users/ahara/AppData/Local/Continuum/miniconda3/envs/dcpython/digital-curling/named/logs/namedLogsVer2.csv"
    with open(file, 'w') as f:
        f.write("")
    """
    preboard(32), ta,sh,w, a, p, nextBoard(32),ta
    """

    df = pd.read_csv("../logs/allLogsVer2NamedShot.csv", header=None)
    for line in tqdm(range(len(df))):
        for _ in range(3):
            turn = random.randint(1, 15)
            prb = []
            for i in range(32):
                prb.append(df.iloc[line, i])
            ta = df.iloc[line, 32]
            vec = getVector(prb, ta, turn % 2)
            if vec != "11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111":
                sh = df.iloc[line, 33]
                w = df.iloc[line, 34]
                a = df.iloc[line, 35]
                p = df.iloc[line, 36]
                afb = []
                for i in range(32):
                    afb.append(df.iloc[line, 37+i])
                score = getScore(afb, turn) - getScore(prb, turn)
                ans = str(vec)+","+str(w)+","+str(a)+","+str(p)+"," + \
                    str(sh)+","+str(turn)+","+str(score)+"\n"
                with open(file, 'a') as f:
                    f.write(ans)

    df = pd.read_csv(file, header=None)
    df = df.drop_duplicates()
    df.to_csv(file, header=False, index=False)


main()
