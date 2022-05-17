# 必须要把根目录的路径添加进来！
import sys
sys.path.append('/home/zp/zpinup/')   
from alogos.HRL.H_DQN.hdqn_base import hdqn_base
import numpy as np
import gym
import torch

def int_to_onehot(input_obs, n=500):
    '''
    把0-499的整数转换为 [500,]的onehot
    '''
    onehot = np.zeros(n)
    onehot[input_obs] = 1.0
    return onehot

def int_to_array(input_obs):
    '''
    把0-499的整数转换为 [1,] 的array
    '''
    obs = np.array([input_obs])
    return obs
 
class h_dqn:
    def __init__(self, env_fn, ac_kwargs=dict(), 
                 buffer_size_meta=int(4e4), buffer_size=int(4e4),
                 gamma=0.99, epsilon=0.0,
                 pi_lr=1e-3, delay_up=0.995, 
                 num_test_episodes=10, max_ep_len=200,
                 device=None):

        self.device = device
        self.env, self.test_env = env_fn(), env_fn()
        # 如果把obs转换成 onehot形式，那么网络的输入维度就是 500 和 1000
        # self.obs_dim = self.env.observation_space.n
        # 如果把obs原生输入，那么网络的输入维度就是 1 和 2
        self.obs_real_dim = self.env.observation_space.n  # 500
        self.obs_dim = 1   # 因为离散的动作，出来肯定是一个单独的整数阿！ 
        self.act_dim = self.env.action_space.n            # 6

        self.meta_controller = hdqn_base(env_fn, self.obs_dim, self.obs_real_dim ,          # 输入是obs，1   输出是goal，维度500
                                  ac_kwargs=ac_kwargs, replay_size=buffer_size_meta,
                                  gamma=gamma, epsilon=epsilon, pi_lr=pi_lr, 
                                  delay_up=delay_up, num_test_episodes=num_test_episodes,
                                  max_ep_len=max_ep_len, device=self.device)
        
        self.controller =  hdqn_base(env_fn, self.obs_dim*2, self.act_dim,         # 输入是obs+goal 维度是1000，输出是action 维度是6，
                                  ac_kwargs=ac_kwargs, replay_size=buffer_size,
                                  gamma=gamma, epsilon=epsilon, pi_lr=pi_lr, 
                                  delay_up=delay_up, num_test_episodes=num_test_episodes,
                                  max_ep_len=max_ep_len, device=self.device)

if __name__ == '__main__':
    env = gym.make('Taxi-v3')
    agent = h_dqn(lambda : env, ac_kwargs=dict(hidden_sizes=[64]*2), device = torch.device('cuda:0'))
    print(agent.meta_controller.policy.pi)
    print(agent.controller.policy.pi)

    obs = env.reset()
    obs = int_to_array(obs)
    # obs = int_to_onehot(obs)

    for i in range(100):
        env.render()
        sub_goal = agent.meta_controller.get_action(obs)
        sub_goal = int_to_array(sub_goal)
        # sub_goal = int_to_onehot(sub_goal)

        state = np.concatenate((obs, sub_goal))
        action = agent.controller.get_action(state)

        obs, r, d, _ = env.step(action)
        obs = int_to_array(obs)
        # obs = int_to_onehot(obs)
        if d:
            break

