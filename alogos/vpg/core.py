import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

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

def discount_cumsum(x, discount):
    '''
    magic from rllab for computing discounted cumulative sums of vectors.就是给rewards计算Gt
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,    
         x1 + discount * x2,                      
         x2]                                      

        Output_t = \sum_{l=0}^inf [(discount)^l]*Input_t+l

    list[::-1]会返回list的倒叙列表，-1是步长
    '''
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

# Actor的基础类以及离散Actor和连续Actor类。
class Actor(nn.Module):
    '''
    创建actor
    类里面的方法出现前置下划线是，代表这个函数是该类私有的，只能在内部调用
    这个类没有 __init__(self, inputs) 所以是不可实例化的类，只是一个用来继承的模板
    '''
    def _distribution(self, obs):
        '''
        这个是提示目前这个函数还没写，是一种技巧，先需要有一个这个函数，另一个类继承过来的时候再写
        obs的维度是[N,obs_dim]，N可以是1，这是就是单个的obs
        如在连续空间中，actor将产生[N,act_dim]维度的mu
        然后利用生成的参数产生分布dist，格式是dist(loc:size=[N,act_dim],scale:size=[N,act_dim])
        dist分布其实就是 pi(.|s)给定s时的分布函数
        '''
        raise NotImplementedError
    
    def _log_prob_from_distribution(self, pi, act):
        '''
        计算 dist.log_prob(a)
        '''
        raise NotImplementedError

    def forward(self, obs, act=None):
        '''
        直接使用forward的情况是在update里面，操作的都是批量数据
        其他调用了forward的函数比如 ac.step都是只接受一个obs
        带梯度
        产生给定状态的分布dist
        计算分布下，给定动作对应的log p(a)
        actor里面forward一般是 只接收批量 的数据，每一步的计算用上面的函数
        '''
        dist = self._distribution(obs)   # \pi(\cdot|s)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(dist, act)
        return dist, logp_a
    
class MLPCategoricalActor(Actor):
    '''
    继承Actor类，并修改一些基类的方法，产生离散的分布，就是PMF概率质量分布率，用于处理离散动作空间 Discrete
    可以实例化
    '''
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        '''初始一个logits网络，可以直接输出各个动作对应的概率'''
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        '''返回离散分布dist [N,act_dim]，每个分布中，每个动作就对应一个确切的概率'''
        logits = self.logits_net(obs)
        return Categorical(logits=logits)
    
    def _log_prob_from_distribution(self, pi, act):
        '''输出形为[N,]的logprob'''
        return pi.log_prob(act) # 离散动作空间，输入act的维度是[N,]，因为选择出来的动作是act_dim里面的一个概率最大的动作,然后输出也是[N,]
                                # 比如倒立摆小车，离散动作空间维度为2，但是最后输出的动作是左或者右，只有1维，这是离散动作的特点！
                                # 输入1个动作，那就输出1个这个动作对应的概率！


class MLPGaussianActor(Actor):
    '''
    继承Actor类，并修改一些基类的方法，产生分布类型是高斯分布，产生的分布是PDF，用于处理连续动作空间 Box
    可以实例化，输入inputs如下，其中hidden_sizes是隐藏层各层的神经元数量数组或者列表,如[100,100]
    再次注意，列表加和还是列表
    '''
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        '''
        高斯actor只有一个mu神经网络
        而高斯actor的log_std也就是log sigma^2不需要由神经网络输出，直接单独作为训练参数
        具体是先产生和动作维度一样的初值，再把这组数变成可训练参数
        '''
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std)) 
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) +[act_dim], activation)

    def _distribution(self, obs):
        '''
        高斯分布
        '''
        mu = self.mu_net(obs)         # mu的形状为 [N, act_dim]
        std = torch.exp(self.log_std) # log_std的形状为[act_dim]
        return Normal(mu, std)        # 虽然形状不匹配，但是可以产生[N, act_dim]的分布，log_std可以广播
    
    def _log_prob_from_distribution(self, pi, act):
        '''输出形为[N,]的logprob'''
        return pi.log_prob(act).sum(axis=-1)    # 最后一维的和, 因为输入act是[N, act_dim],需要返回形状为[N,]的求和结果
                                                # 连续动作空间，比如多关节环境，动作的维度为3，那么每次输出的动作维度也是3维，这是连续动作区间的特点
                                                # 比如说一个动作是[1.1,0.1,2.0]，计算出来的dist.logprob是(-1,-1,-2)，虽然是三维，但是毕竟这就只是一个动作
                                                # 因此函数_log_prob_from_distribution计算出来的是这一个动作的log概率=-4
                                                # 输入1个(组)动作，得到1个这个(组)动作对应的概率.



# Critic 只有一个基础MLP类，不需要基础类，直接一个可以实例化的类就行了。
class MLPCritic(nn.Module):
    '''Critic的输出维度只有[N,1]，输入是状态'''
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
    
    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # 保证Critic输出的价值的维度也是 [N,]

# 把Actor和Critic合并成一类
class MLPActorCritic(nn.Module):
    '''
    创建一个默认参数的，可以调用的ActorCritic网络
    '''
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()
        
        obs_dim = observation_space.shape[0]

        # 根据环境是离散动作区间还是连续动作区间来建立对应的Actor策略pi
        if isinstance(action_space, Box): # 如果动作区间的类是Box，也就是连续动作空间
            act_dim = action_space.shape[0]
            self.pi = MLPGaussianActor(obs_dim, act_dim, hidden_sizes, activation)
        elif isinstance(action_space, Discrete): # 如果动作空间类是 Discrete，也就是离散动作空间
            act_dim = action_space.n
            self.pi = MLPCategoricalActor(obs_dim, act_dim, hidden_sizes, activation)
        
        # 建立Critic策略v
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        '''
        step仅仅接受 1个obs ，用于驱动环境运行，并记录old的各种量
        不用梯度，测试的输出该状态下
        使用策略得到的动作， 状态的价值， 动作对应的log p(a)
        '''
        with torch.no_grad():
            dist = self.pi._distribution(obs)
            a = dist.sample()
            logp_a = self.pi._log_prob_from_distribution(dist, a)

            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()
    
    def act(self, obs):
        '''
        这个函数，仅仅用在ppo_test里面，给一个状态，得到一个动作，用于测试。
        '''
        return self.step(obs)[0]



if __name__ == '__main__':
    import gym
    env = gym.make('Hopper-v2')
    ac = MLPActorCritic(env.observation_space,env.action_space)
    obs = env.reset()
    print(ac.step(torch.FloatTensor(obs)))



    
