import pandas as pd
import random
import math
powerData = [3,5,7,12,16]
action = random.randint(0,29)
power = math.floor(action/6)
action2 = action-power*6
angle = math.floor(action2/3)
where = action2-angle*3
df = pd.read_csv('./logs.csv', sep=',')
df = df[(df['vec']==900000000100)&(df['where']==where)&(df['angle']==angle)&(df['power']==powerData[power])]
print(df)
print("ac=",action)
print("wh=",where)
print("an=",angle)
print("po=",power)
print("min=",df['reward'].min())
print("max=",df['reward'].max())
print("mean=",df['reward'].mean())