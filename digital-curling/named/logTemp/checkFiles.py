import pandas as pd
df = pd.read_csv("./testLogs.csv", sep=',', header=None, names=(
    'vector', 'shot', 'reward'))
a = []
for i in range(8):
    ddf = df[df['shot'] == i]
    mean = ddf['reward'].mean()
    print(mean)
    a.append(mean)
    dff = ddf[ddf['reward'] > mean]
    print(dff['reward'].mean())
    print(len(ddf))

    print(len(dff))
print(a)
