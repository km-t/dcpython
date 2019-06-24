import pandas as pd
from tqdm import tqdm


def getVecs():
    d = pd.read_csv('../logs/vectorsVer2.csv', header=None)
    vecs = []
    for i in range(len(d)):
        vecs.append(d.iloc[i, 0])
    return vecs


def initFile():
    with open("../logs/highestScoreLogs.csv", 'w') as f:
        pass


def getDF(df, vec):
    dff = df[df['vector'] == vec]
    dff = dff.sort_values(by=["reward"], ascending=False)
    return dff


def writeFile():
    vecs = getVecs()
    df = pd.read_csv('../logs/highScoreLogsVer2.csv', sep=',', header=None, names=(
        'vector', 'where', 'angle', 'power', 'reward'))
    for vec in tqdm(vecs):
        dff = getDF(df, vec)
        if dff.empty:
            print("")
            print(vec)
        else:
            val = ""
            for j in range(5):
                val += str(dff.iloc[0, j])
                if j < 4:
                    val += ","
                else:
                    val += "\n"
            with open("../logs/highestScoreLogs.csv", 'a')as f:
                f.write(val)


print("out high score log")
initFile()
writeFile()
