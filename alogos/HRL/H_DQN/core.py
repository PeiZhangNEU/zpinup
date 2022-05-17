import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F

class Reshape(nn.Module):
    '''
    reshpe层加到Sequential可以
    '''
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((self.shape))

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

    # 这个回头肯定还要改，因为是针对Taxi环境的！
    layers = []
    layers.append(nn.Embedding(500, 10))   # 针对taxi环境，把介于0-499之间的整数密集映射到10维

    if sizes[0] == 1:
        layers.append(Reshape(-1,10))   # 把sizes的第0维换成10, 不太行阿，输入是2个整数的时候，就映射到20了阿！
        sizes[0] = 10
    elif sizes[0] == 2:
        layers.append(Reshape(-1,20))
        sizes[0] = 20                   # 【N,2】的输入embedding之后是[N, 2,10], 变成[N,10]最后[N, 6]
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    '''
    返回一个模型所有的参数数量
    '''
    return sum([np.prod(p.shape) for p in module.parameters()])

# Actor的基础类以及离散Actor和连续Actor类。
class Q_net(nn.Module):
    '''
    创建actor
    类里面的方法出现前置下划线是，代表这个函数是该类私有的，只能在内部调用
    这个类没有 __init__(self, inputs) 所以是不可实例化的类，只是一个用来继承的模板
    '''
    def __init__(self, input_dim, output_dim, hidden_sizes, activation):
        '''初始一个logits网络，可以直接输出各个动作对应的概率, 默认useconve=False, 这样外部的类就不用加东西了，只有训练atari的时候再加就好了'''
        super().__init__()
       
        self.logits_net = mlp([input_dim] + list(hidden_sizes) + [output_dim], activation) 
    
    def forward(self, obs):
        '''
        这个函数是为了计算目前的logpa，操作的是批量数据，批量数据仅仅在update的时候需要用到！
        只在upadate这一步计算loss时才需要用到
        带梯度
        产生给定状态的分布dist
        计算分布下，给定动作对应的log p(a)
        actor里面forward一般是只接收批量的数据，每一步的计算用上面的函数
        '''
        logits = self.logits_net(obs)
        return logits

# 把网络整合到一个agent中
class MLPpolicy(nn.Module):
    '''
    创建一个默认参数的，可以调用的ActorCritic网络
    DQN 只能处理离散动作，所以act_dim 直接取n
    '''
    def __init__(self, in_dim, out_dim, epsilon=0.5,
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()
        self.epsilon = epsilon
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pi = Q_net(self.in_dim, self.out_dim, hidden_sizes, activation)
        
    def step(self, obs):
        '''
        只接受1个obs，用于驱动环境运行
        贪婪策略
        在训练时，动作用这个产生, 这个过程本身就是贪婪策略了，在train的时候直接用即可
        '''
        if np.random.uniform() >= self.epsilon:
            with torch.no_grad():
                logits = self.pi(obs).cpu().numpy()
                a = np.argmax(logits)               # 无论logtis的形状是[1,4] 还是[4,] argmax得到的结果一样的！
        else:
            a = np.random.randint(0, self.out_dim)

        return a
    
    def act(self, obs):
        '''用于载入模型之后测试'''
        with torch.no_grad():
            logits = self.pi(obs).cpu().numpy()
            a = np.argmax(logits)
        return a
    




    
