import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('/home/zp/zpinup/')
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch

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

def conv_mlp(obs_dim, act_dim):
    '''
    对于atari环境，输入是图片的情况来说，需要用到卷积处理图片，方便起见，我这里的网络结构直接制定了
    obs_dim = [1, 84, 84]
    '''
    # 先建立一个卷及模型
    conv_model = nn.Sequential(
                nn.Conv2d(obs_dim[0], 32, 8, 4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1),
                nn.ReLU()
            )
    # 再进行全连接, 全连接之前必须知道卷积的输出拉直之后是什么形状
    tmp = torch.zeros(1, * obs_dim)
    feature_size = conv_model(tmp).view(1, -1).size(1)

    # 再建立一个线性模型
    linear_model = nn.Sequential(
                nn.Linear(feature_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, act_dim)
            )

    return conv_model, linear_model

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
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, useconve=False):
        '''初始一个logits网络，可以直接输出各个动作对应的概率, 默认useconve=False, 这样外部的类就不用加东西了，只有训练atari的时候再加就好了'''
        super().__init__()
        self.useconve = useconve
        if self.useconve:
            # 如果使用卷及网络，那就用conv层
            self.conv_net, self.logits_net = conv_mlp(obs_dim, act_dim)
        else:
            self.logits_net = mlp([obs_dim[0]] + list(hidden_sizes) + [act_dim], activation) # 把obs_dim[0] 防在这里是最合适的
    
    def forward(self, obs):
        '''
        这个函数是为了计算目前的logpa，操作的是批量数据，批量数据仅仅在update的时候需要用到！
        只在upadate这一步计算loss时才需要用到
        带梯度
        产生给定状态的分布dist
        计算分布下，给定动作对应的log p(a)
        actor里面forward一般是只接收批量的数据，每一步的计算用上面的函数
        '''
        if self.useconve:
            conv_feature = self.conv_net(obs).view(obs.size(0), -1)   # [N, conv_features num]
            logits = self.logits_net(conv_feature)
        else:
            logits = self.logits_net(obs)
        return logits



# 把网络整合到一个agent中
class MLPpolicy(nn.Module):
    '''
    创建一个默认参数的，可以调用的ActorCritic网络
    DQN 只能处理离散动作，所以act_dim 直接取n
    '''
    def __init__(self, observation_space, action_space, epsilon=0.5,
                 hidden_sizes=(64,64), activation=nn.Tanh, useconve=False):
        super().__init__()
        self.obs_dim = observation_space.shape    # 对于非atari环境来说，obsshape=[obs_dim,], 对于atari来说， obsshape=[1,84,84]
        self.act_dim = action_space.n
        self.epsilon = epsilon
        self.pi = Q_net(self.obs_dim, self.act_dim, hidden_sizes, activation, useconve=useconve)
        
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
            a = np.random.randint(0, self.act_dim)

        return a
    
    def act(self, obs):
        '''用于载入模型之后测试'''
        with torch.no_grad():
            logits = self.pi(obs).cpu().numpy()
            a = np.argmax(logits)
        return a
    

if __name__ == '__main__':
    import gym
    # env = gym.make('CartPole-v1')
    env = gym.make('Breakout-v0', render_mode='human')  # atari游戏在这里设置完rendermode之后在下面不需要再 env.render()
    ## atari game env原本的obs形状为不固定的比如 [210, 160, 3] ，但pytorch的图表格式channel在最前 
    ## 所以使用deepmind开放的wrap可以把atari env 解构，然后把 obs 形状转换为一样的形状 [1, 84, 84]， 在对atari环境进行卷及网络的时候一定要首先对环境wrapper
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)
    ac = MLPpolicy(env.observation_space,env.action_space, useconve=True)
    obs = env.reset()
    for i in range(200):
        # env.render()
        a = ac.step(torch.FloatTensor(obs))
        obs, r, d, _ = env.step(a)
        if d:
            break




    
