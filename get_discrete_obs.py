import gym
import numpy as np

def int_to_onehot(input_obs, n=500):
    '''
    把0-499的整数转换为 1*500的onehot
    '''
    onehot = np.zeros(n)
    onehot[input_obs] = 1.0
    return onehot


env = gym.make('Taxi-v3')  

obs = env.reset()
obs = int_to_onehot(obs)

while True:
    env.render()
    a = env.action_space.sample()
    obs, r, d,_ = env.step(a)
    obs = int_to_onehot(obs)
    if d:
        break
