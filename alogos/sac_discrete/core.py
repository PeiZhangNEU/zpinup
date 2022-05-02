
import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


# 几个全局函数

def combined_shape(length, shape=None):
    '''
    把length和shape结合起来得到列表，用于buffer的存储数据的形状初始化.
    比如Discrete环境的 actdim=(), 10，()就是得到[10,]
    例如：10,[100,3] 得到 [10,100,3]
         10, 3 得到 [10,3]
         10, None 得到 [10,]
         return A If B else C代表，如果B，返回A，否则返回C
         return *列表是返回列表里面的值，例如return *[1,2,3] = 1,2,3
    '''
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    '''
    创建一个mlp序列神经网络模型
    sizes是一个列表，是每层的神经元个数,包括输入和输出层的神经元个数, 如 [10, 100, 3]
    两个列表直接相加会得到一个新的列表： [1,2,3] + [3,4] = [1,2,3,3,4]
    nn.Identity这个激活函数代表不加任何激活，直接输出，也就是说默认输出层没有激活函数
    '''
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    '''
    返回一个模型所有的参数数量
    '''
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPActor(nn.Module):
    '''
    category 策略，输入obs，输出 act pi(s) logpi(s)
    '''
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.net = mlp([obs_dim] + list(hidden_sizes) +[act_dim], activation)  # 输出不加激活直接输出logits

    def forward(self, obs, deterministic=False, with_logprob=True):
        logits = self.net(obs)
        
        # 产生离散分布dist
        pi_distribution = Categorical(logits=logits)
        # 得到各个动作对应的概率, 这其实是把 logits 做 softmax后的结果
        probs = pi_distribution.probs

        if deterministic:
            # 确定性选取动作，直接查表法找最大概率对应的动作 [N, 1]一定是一个动作，因为是离散空间
            pi_action = torch.argmax(probs.view(-1, self.act_dim), dim=1).squeeze(-1)
        else:
            pi_action = pi_distribution.sample()   # 必须要用 带梯度的 rasmple！ 因为我们优化策略的时候需要求 \grad Q(s, \tilde a_\theta(s))! 

        # 不再计算 ln(pi(a|s)) 而是 计算所有动作的概率 ln(pi(s))
        if with_logprob:
            logp_pi = pi_distribution.logits  # ln(pi(s))  【N，act_dim】 Category分布的logits就是log probs
            
        else:
            logp_pi = None

        return pi_action, probs, logp_pi


class MLPQFunction(nn.Module):
    ''' 离散形的Q网络，输入s，输出a维度的价值'''

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)   # 输出不加激活

    def forward(self, obs):
        q = self.q(obs)
        return q

class MLPActorCritic(nn.Module):
    '''
    温度参数直接加到sac里面，不再这里放！
    '''

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.n              # 离散空间的act_dim = n

        # 建立一个策略网络和两个q网络
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ , _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()


if __name__ == '__main__':
    import gym
    env = gym.make('CartPole-v0')
    ac = MLPActorCritic(env.observation_space,env.action_space)
    obs = env.reset()
    a = ac.act(torch.FloatTensor(obs), True)
    print(a)
    obs2, r, d, _ = env.step(a)
    obs_data = torch.as_tensor([obs, obs2])
    print(ac.pi(obs_data, True))



    
