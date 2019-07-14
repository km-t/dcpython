import pandas as pd
from tqdm import tqdm


def initFile():
    with open("../logs/balancedPowerLogs.csv", 'w') as f:
        pass


def getDF(df, power):
    dff = df[df['power'] == power]
    dff = dff.sort_values(by=["reward"], ascending=False)
    return dff


def writeFile(file):
    df = pd.read_csv(file, sep=',', header=None, names=(
        'vector', 'where', 'angle', 'power', 'reward', 'aa'))
    df = df[df['reward'] > 0]
    df = df[df['vector'] !=
            '11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111']
    powers = [4, 6, 8, 10, 12]
    ddd = df
    vecs = []
    for power in powers:

        df = df[~df['vector'].isin(vecs)]
        dff = getDF(df, power)
        # dff = dff[~dff['vector'].duplicated()]
        duplicatedSize = min(20000, len(dff))
        print(min(20000, len(dff)))
        for i in range(duplicatedSize):
            vecs.append(str(dff.iloc[i, 0]))
            val = ""
            for j in range(6):
                val += str(dff.iloc[i, j])
                if j < 5:
                    val += ","
                else:
                    val += "\n"
            with open("../logs/balancedPowerLogs.csv", 'a')as f:
                f.write(val)


def main(file):
    print("out balanced power score log")
    initFile()
    writeFile(file)


main("../logs/namedLogs.csv")
