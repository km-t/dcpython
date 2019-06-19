import pandas as pd
from tqdm import tqdm


def getVecs():
    d = pd.read_csv('../logs/vectorsVer2.csv', header=None)
    vecs = []
    for i in range(len(d)):
        vecs.append(d.iloc[i, 0])
    return vecs


def initFile():
    with open("../logs/highScoreLogsVer2.csv", 'w') as f:
        pass


def getDF(df, vec):
    dff = df[df['vector'] == vec]
    dff = dff.sort_values(by=["reward"], ascending=False)
    return dff


def writeFile():
    vecs = getVecs()
    df = pd.read_csv('../logs/logsVer2.csv', sep=',', header=None, names=(
        'vector', 'where', 'angle', 'power', 'reward'))
    df = df[df['reward'] > 0]
    for vec in tqdm(vecs):
        dff = getDF(df, vec)
        duplicatedSize = min(5, len(dff))
        for i in range(duplicatedSize):
            val = ""
            for j in range(5):
                val += str(dff.iloc[i, j])
                if j < 4:
                    val += ","
                else:
                    val += "\n"
            with open("../logs/highScoreLogsVer2.csv", 'a')as f:
                f.write(val)


def main():
    print("out high score log")
    initFile()
    writeFile()
