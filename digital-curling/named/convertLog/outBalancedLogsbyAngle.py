import pandas as pd
from tqdm import tqdm


def initFile():
    with open('../logs/balancedAngleLogs.csv', 'w') as f:
        pass


def getDF(df, angle):
    dff = df[df['angle'] == angle]
    dff = dff.sort_values(by=["reward"], ascending=False)
    return dff


def writeFile(file):
    df = pd.read_csv(file, sep=',', header=None, names=(
        'vector', 'where', 'angle', 'power', 'reward', 'aa'))
    df = df[df['reward'] > 0]
    df = df[df['vector'] !=
            '11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111']
    angles = [0, 1]
    ddd = df
    vecs = []
    for angle in angles:
        df = df[~df['vector'].isin(vecs)]
        dff = getDF(df, angle)
        duplicatedSize = min(50000, len(dff))
        print(min(50000, len(dff)))
        for i in range(duplicatedSize):
            vecs.append(str(dff.iloc[i, 0]))
            val = ""
            for j in range(6):
                val += str(dff.iloc[i, j])
                if j < 5:
                    val += ","
                else:
                    val += "\n"
            with open('../logs/balancedAngleLogs.csv', 'a')as f:
                f.write(val)


def main(file):
    print("out balanced angle score log")
    initFile()
    writeFile(file)
