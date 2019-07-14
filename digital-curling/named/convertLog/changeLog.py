import pandas as pd
from tqdm import tqdm


def initFile():
    with open('../logs/balancedLogs.csv', 'w') as f:
        pass


def getDF(df, shot):
    dff = df[df['shot'] == shot]
    dff = dff.sort_values(by=["reward"], ascending=False)
    return dff


def writeFile(file):
    df = pd.read_csv(file, sep=',', header=None, names=(
        'vector', 'where', 'angle', 'power', 'reward', 'shot'))
    df = df[df['reward'] > 0]
    df = df[df['vector'] !=
            '11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111']
    shots = [3, 2, 7, 4, 5, 0, 6, 1]
    vecs = []
    ddd = df
    for shot in shots:
        df = df[~df['vector'].isin(vecs)]
        dff = getDF(df, shot)
        duplicatedSize = min(7000, len(dff))
        print(min(7000, len(dff)))
        for i in range(duplicatedSize):
            vecs.append(str(dff.iloc[i, 0]))
            val = ""
            for j in range(6):
                val += str(dff.iloc[i, j])
                if j < 5:
                    val += ","
                else:
                    val += "\n"
            with open('../logs/balancedLogs.csv', 'a')as f:
                f.write(val)


def main(file):
    initFile()
    writeFile(file)
