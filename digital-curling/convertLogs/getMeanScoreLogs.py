import pandas as pd
from tqdm import tqdm


def initFile():
    with open('../logs/meanScoreLogs.csv', 'w') as f:
        pass


def getDF(df, angle):
    dff = df[df['angle'] == angle]
    dff = dff.sort_values(by=["reward"], ascending=False)
    return dff


def writeFile(file):
    df = pd.read_csv(file, sep=',', header=None, names=(
        'vector', 'where', 'angle', 'power', 'reward', 'turn'))
    angles = [0, 1]
    where = [0, 1, 2]
    power = [3, 5, 7, 12, 16]
    vecs = []
    dff = df[~df['vector'].isin(vecs)]
    hoge = dff['vector']
    vecSize = hoge.duplicated().value_counts()[False]
    for _ in tqdm(range(vecSize)):
        dff = df[~df['vector'].isin(vecs)]
        v = str(dff.iloc[0, 0])
        vecs.append(v)
        ans = ""
        for w in where:
            for a in angles:
                for p in power:
                    d = dff[dff['vector'] == v]
                    d = d[d['where'] == w]
                    d = d[d['angle'] == a]
                    d = d[d['power'] == p]
                    turn = []
                    turn.append(int(d.iloc[0, 5]) % 16)
                    for i in range(len(d)):
                        for j in range(len(turn)):
                            if turn[j] == int(d.iloc[i, 5]) % 16:
                                pass
                            else:
                                turn.append(int(d.iloc[i, 5]) % 16)
                    print(turn)
                    exit()


"""
                    for t in turn:
                        d = d[d['turn'] == t]
                        score = 0
                        size = len(d)
                        for i in range(size):
                            score += float(d.iloc[i, 4])
                        score /= size
                        ans += v+","+str(w)+","+str(a)+"," + \
                            str(p)+","+str(score)+','str(t)+"\n"
"""
"""
        with open('../logs/meanScoreLogs.csv', 'a')as f:
            f.write(ans)
"""

initFile()
writeFile("../logs/logsVer2Ver2.csv")
