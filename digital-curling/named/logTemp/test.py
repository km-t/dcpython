import pandas as pd
from tqdm import tqdm


def getVecSize(df, shot):
    d = df['vector']
    df = df[df['shot'] == shot]
    dupSize = d.duplicated().value_counts()[False]
    return dupSize


df = pd.read_csv("./namedLogs.csv", sep=',', header=None, names=(
    'vector', 'where', 'angle', 'power', 'reward', 'shot'))
df = df[df['vector'] !=
        '11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111']
shots = [0, 1, 2, 3, 4, 5, 6, 7]
for shot in shots:
    dupSize = getVecSize(df, shot)
    dff = df[df['shot'] == shot]
    flag = True
    vecs = []
    while flag:
        dff = dff[~dff['vector'].isin(vecs)]
        if len(dff) > 0:
            vec = str(dff.iloc[0, 0])
            vecs.append(vec)
            val = str(dff.iloc[0, 0])+","+str(shot)+"," + \
                str(dff[dff['vector'] == vec]['reward'].mean())+"\n"
            with open("./testLogs.csv", 'a')as f:
                f.write(val)
        else:
            flag = False
