import sys
import gym
import numpy as np
import gym.spaces
import math
import pandas as pd

df = pd.read_csv('./logs.csv', sep=',')
df = df.sample(frac=1)


def getData(line, keyNum):
    if keyNum == 0:  # vec
        vec = str(df.iloc[line, 0])
        v = np.zeros(11, dtype=np.float32)
        for i in range(11):
            v[i] = float(vec[i+1])
        return v
    else:  # where, angle, power, reward
        if keyNum==4:
            ans=df.iloc[line+1, keyNum]
        else:
            ans = df.iloc[line, keyNum]
        return ans


class MyEnv(gym.core.Env):
    def __init__(self):
        self.board = np.zeros(11, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(30)
        low_bound = 0
        high_bound = 1
        self.observation_space = gym.spaces.Box(
            low=low_bound, high=high_bound, shape=self.board.shape, dtype=np.float32)
        self.time = 0
        self.obs = getData(0,0)

    def step(self, action):
        st = "9"
        for i in range(len(observation)):
            st+=str(int(observation[i]))

        power = math.floor(action/6)
        action = action-power*6
        angle = math.floor(action/3)
        action = action-angle*3
        where = action

        df2 = df[(df['vec']==st)&(df['where']==where)&(df['angle']==angle)&(df['power']==power)]
        df2 = df2.sample(frac=1)

        reward = float(df2.iloc[0,4])
        self.time+=1
        observation = getData(self.time, 0)
        done = True
        return observation, reward, done, {}

    def reset(self):
        self.obs = getData(self.time, 0)