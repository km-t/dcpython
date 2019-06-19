import pandas as pd
from tqdm import tqdm


def main():
    print("out vector")
    df = pd.read_csv('../logs/logsVer2.csv', sep=',', header=None, names=(
        'vector', 'where', 'angle', 'power', 'reward'))

    vecs = []
    vecs.append(df.iloc[0, 0]+"\n")
    for i in tqdm(range(len(df))):
        flag = True
        for vec in vecs:
            if vec == df.iloc[i, 0]+"\n":
                flag = False
                break
        if flag:
            vecs.append(df.iloc[i, 0]+"\n")
    with open("../logs/vectorsVer2.csv", 'w') as f:
        for vec in vecs:
            f.write(vec)
