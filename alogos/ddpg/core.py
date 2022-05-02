import numpy as np
import scipy.signal

import torch
import torch.nn as nn

# DDPG只能进行连续动作区间的环境处理，不需要引入离散动作区间环境的网络
# 几个全局函数
def get_mean_and_std(x):
    '''计算一组数据的均值和方差'''
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = np.sum(x), len(x)
    mean = global_sum / global_n

    global_sum_sq = np.sum((x - mean)**2)
    std = np.sqrt(global_sum_sq / global_n)  # compute global std
    return mean, std

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
    '''简单的确定性输出策略'''
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)         # 输出层+ Tanh 映射到-1，1
        self.act_limit = act_limit
    
    def forward(self, obs):
        '''返回符合环境动作范围的动作
        直接使用的时候，仅仅在update函数使用，操作批量数据！
        '''
        return self.act_limit * self.pi(obs)

class MLPQFunction(nn.Module):
    '''简单的动作价值网络'''
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)   # 输出层不加激活

    def forward(self, obs, act):
        '''返回动作价值'''
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)   # 让q的形状必须是 (N,)

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256), activation=nn.ReLU):
        super().__init__() 

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # 建造俩网络
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
    
    def act(self, obs):
        '''根据1个obs选择确定性动作，仅用于get action函数，单步动作驱动环境运行！'''
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()
        

if __name__ == '__main__':
    import gym
    env = gym.make('Hopper-v2')
    ac = MLPActorCritic(env.observation_space,env.action_space)
    obs = env.reset()
    print(ac.step(torch.FloatTensor(obs)))



    
