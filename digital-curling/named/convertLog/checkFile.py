import pandas as pd
patha = "../logs/"
pathb = ['balancedWhereLogs', 'balancedAngleLogs',
         'balancedPowerLogs', 'balancedLogs']
wheres = [0, 1, 2]
angles = [0, 1]
powers = [4, 6, 8, 10, 12]
shot = [0, 1, 2, 3, 4, 5, 6, 7]
logs = [wheres, angles, powers, shot]
names = ['where', 'angle', 'power', 'shot']
for i in range(len(pathb)):
    path = patha+pathb[i]+'.csv'
    df = pd.read_csv(path, sep=',', header=None, names=(
        'vector', 'where', 'angle', 'power', 'reward', 'shot'))
    log = logs[i]
    for j in log:
        dff = df[df[names[i]] == j]
        print(names[i], j)
        print(len(dff))
