import pandas as pd
from tqdm import tqdm


def initFile():
    with open("../logs/log0.csv", 'w') as f:
        pass


def getDF(df, where):
    dff = df[df['where'] == where]
    dff = dff.sort_values(by=["reward"], ascending=False)
    return dff


def writeFile(file):
    df = pd.read_csv(file, sep=',', header=None, names=(
        'vector', 'where', 'angle', 'power', 'shot', 'reward'))

    df = df[df['where'] != -1]
    vecs = []
    wheres = [0, 1, 2]
    for where in wheres:
        df = df[~df['vector'].isin(vecs)]
        dff = getDF(df, where)
        # dff = dff[~dff['vector'].duplicated()]
        duplicatedSize = min(33333, len(dff))
        print(min(33333, len(dff)))
        for i in range(duplicatedSize):
            vecs.append(str(dff.iloc[i, 0]))
            val = ""
            for j in range(6):
                val += str(dff.iloc[i, j])
                if j < 5:
                    val += ","
                else:
                    val += "\n"
            with open("../logs/log0.csv", 'a')as f:
                f.write(val)


def main(file):
    print("out balanced where score log")
    initFile()
    writeFile(file)


main("../logs/namedLogsWithTurn.csv")
