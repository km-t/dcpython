import winsound
import time
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np
import myenv
import random
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


env = gym.make("myenv-v0")

obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n


class RandomActor:
    def __init__(self):
        pass

    def random_action_func(self):
        action = env.getAction(1)
        return action


class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=200):
        super().__init__(
            l0=L.Linear(obs_size, n_hidden_channels),
            l1=L.Linear(n_hidden_channels, n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_actions))

    def __call__(self, x, test=False):
        """
        x ; 観測#ここの観測って、stateとaction両方？
        test : テストモードかどうかのフラグ
        """
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))


q_func = QFunction(obs_size, n_actions)

optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)
gamma = 0.95

ra = RandomActor()
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=1, random_action_func=ra.random_action_func)
#epsilon=1で必ずra.random_action_funcが選ばれる
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10**6)


def phi(x): return x.astype(np.float32, copy=False)


agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500000, update_interval=1,
    target_update_interval=100, phi=phi)


# 学習
_df = pd.read_csv('./logs.csv', sep=',')
n_episodes = len(_df)
start = time.time()
lossData = []
obs = env.getObs()
reward = env.getRew()
for i in range(n_episodes):
    action = agent.act_and_train(obs, reward)  # rewardこれでいいのか？
    obs, reward, done, _ = env.step(action)
    agent.stop_episode_and_train(obs, reward, done)
    lossData.append(agent.get_statistics()[1][1])
    if i % 500 == 0:
        print('episode:', i,
              'R:', reward,
              'statistics:', agent.get_statistics())
print('Finished, elapsed time : {}'.format(time.time()-start))

save_path = "./result/"
agent.save(save_path)

winsound.Beep(1000, 1000)

plt.plot(lossData)
plt.xlabel("No of trails")
plt.ylabel("loss")
plt.title("or.py")
plt.legend()
plt.show()
