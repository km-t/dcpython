import pandas as pd
from tqdm import tqdm


def main():
    print("out vector")
    df = pd.read_csv('../logs/logsVer2.csv', sep=',', header=None, names=(
        'vector', 'where', 'angle', 'power', 'reward'))
    df = df[df['reward'] > 0]
    df = df['vector']
    print(len(df))
    df = df.drop_duplicates()
    print(len(df))
    df.to_csv("../logs/vectorsVer2.csv", sep=",", header=None, index=False)
