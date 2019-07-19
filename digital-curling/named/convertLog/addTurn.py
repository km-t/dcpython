import pandas as pd
from tqdm import tqdm


def initFile(file):
    with open(file, 'w') as f:
        pass


def writeFile(file):
    df = pd.read_csv(file, sep=',', header=None, names=(
        'vector', 'where', 'angle', 'power', 'shot', 'turn', 'reward'))
    df = df[df['reward'] > 0]
    df = df[df['vector'] !=
            "11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111"]
    size = len(df)
    initFile("../logs/namedLogsWithTurn.csv")
    for i in tqdm(range(size)):
        v = str(df.iloc[i, 0])
        w = str(df.iloc[i, 1])
        a = str(df.iloc[i, 2])
        p = str(df.iloc[i, 3])
        s = str(df.iloc[i, 4])
        t = str(df.iloc[i, 5]-1)
        r = str(df.iloc[i, 6])
        turn = "00000000000000"
        if int(t) < 14:
            turn = turn[:int(t)] + "1"+turn[int(t)+1:]
        v += turn
        val = v+","+w+","+a+","+p+","+s+","+r+"\n"
        with open("../logs/namedLogsWithTurn.csv", 'a')as f:
            f.write(val)


def main(file):
    writeFile(file)
