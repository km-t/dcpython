import pandas as pd
from tqdm import tqdm


def initFile():
    with open("../logs/balancedWhereLogs.csv", 'w') as f:
        pass


def getDF(df, where):
    dff = df[df['where'] == where]
    dff = dff.sort_values(by=["reward"], ascending=False)
    return dff


def writeFile(file):
    df = pd.read_csv(file, sep=',', header=None, names=(
        'vector', 'where', 'angle', 'power', 'reward'))
    df = df[df['reward'] > 0]
    df = df[df['vector'] !=
            '11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111']

    vecs = []
    wheres = [0, 1, 2]
    for where in wheres:
        df = df[~df['vector'].isin(vecs)]
        dff = getDF(df, where)
        #dff = dff[~dff['vector'].duplicated()]
        duplicatedSize = min(4500, len(dff))
        for i in range(duplicatedSize):
            vecs.append(str(dff.iloc[i, 0]))
            val = ""
            for j in range(5):
                val += str(dff.iloc[i, j])
                if j < 4:
                    val += ","
                else:
                    val += "\n"
            with open("../logs/balancedWhereLogs.csv", 'a')as f:
                f.write(val)


def checkFile():
    df = pd.read_csv('../logs/balancedWhereLogs.csv', sep=',', header=None, names=(
        'vector', 'where', 'angle', 'power', 'reward'))
    wheres = [0, 1, 2]
    for where in wheres:
        dff = df[df['where'] == where]
        print(len(dff))


def main(file):
    print("out balanced where score log")
    initFile()
    writeFile(file)
    checkFile()
