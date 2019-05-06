import sys
import gym
import numpy as np
import gym.spaces
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


class MyEnv(gym.Env):
    def __init__(self):
        self.board = np.zeros(11, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        low_bound = 0
        high_bound = 1
        self.observation_space = gym.spaces.Box(
            low=low_bound, high=high_bound, shape=self.board.shape, dtype=np.float32)
        self.time = 0
        self.profit = 0

    def step(self, action):

        """
        actionに対するrewardを計算してobservationを返す
        doneは終了かどうか
        """
        reward = getData(self.time, 4)
        self.time += 1
        self.profit += reward
        done = self.time == (len(df) - 1)
        if done:
            print("profit___{}".format(self.profit))
        info = {}
        observation = getData(self.time, 0)
        return observation, reward, done, info

    def reset(self):
        """
        done==trueで実行される
        初期化する
        """
        return getData(0, 0)

    def render(self, mode):
        pass

    def close(self):
        pass

    def seed(self):
        pass

    def getObs(self):
        return getData(self.time, 0)

    def getAction(self, keyNum):
        return getData(self.time, keyNum)
    
    def getRew(self):
        return getData(self.time, 4)