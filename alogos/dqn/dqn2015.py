from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import alogos.dqn.core as core
import torch.nn.functional as F

def Merge(dict1, dict2): 
    '''合并俩字典'''
    res = {**dict1, **dict2} 
    return res 

class ReplayBuffer:
    '''
    一个简单的 first in first out 队列 experience replay buffer
    '''
    def __init__(self, obs_dim, act_dim, size, device=None):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, 1), dtype=np.float32)            # 对于离散环境，a的维度一定是1，因为最终得到的动作是整数！
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size   # self.size 代表目前buffer一共多大，max size代表buffer的最大限度
        self.device = device
    
    def store(self, obs, act, rew, next_obs, done):
        '''
        把每一步的反馈存到buffer里，----------这里就是和 on-policy 策略的区别之一，指针循环---------
        '''
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        # 这里就是和on - policy策略的区别之一： on-policy的ptr指针是不断+1的，直到一个epoch结束，get数据的时候把ptr置0
        # 而off-policy则是固定一个buffer，不断存数据，存满了再从头往后覆盖。
        self.ptr = (self.ptr + 1) % self.max_size     # 指针循环，循环存储，当存满了就从头再存覆盖掉最早的数据。
        self.size = min(self.size + 1, self.max_size) # 目前buff的大小 没存满之前 self.size += 1自加1，到了最大大小就固定了。
    
    def sample_batch(self, batch_size=32):
        '''
        从目前的buffer中随机采集batchsize的数据， ---------这个是和on-policy区别之二，是sample，而不是直接把存满的buffer直接get过来-------------
        '''
        idxs = np.random.randint(0, self.size, batch_size)
        batch = dict(obs=self.obs_buf[idxs], 
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k,v in batch.items()}
    
class dqn:
    '''
    定义了DQN的结构和更新方式
    环境的动作空间一定是离散的
    '''
    def __init__(self, env_fn, policy=core.MLPpolicy, ac_kwargs=dict(), 
                 replay_size=int(1e6), gamma=0.99, epsilon=0.5,
                 pi_lr=1e-3, delay_up=0.995, useconve=False,
                 num_test_episodes=10, max_ep_len=1000, device=None
        ):

        self.device = device
        self.env, self.test_env = env_fn(), env_fn()
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.n
        self.gamma = gamma
        self.delay_up = delay_up
        self.num_test_epsodes = num_test_episodes
        self.max_ep_len = max_ep_len



        # 建立Q_net
        self.policy = policy(self.env.observation_space, self.env.action_space, epsilon=epsilon, useconve=useconve, **ac_kwargs).to(self.device)
        self.pi_optimizer = Adam(self.policy.pi.parameters(), lr=pi_lr)

        # 建立Q_target: 直接复制policy
        self.tar_policy = deepcopy(self.policy)
        # 冻结目标 ac_targ所有的 的参数, 包括q和pi，以减少计算资源
        for p in self.tar_policy.parameters():
            p.requires_grad = False

        
        # 初始buffer
        self.buffer = ReplayBuffer(self.obs_dim, self.act_dim, replay_size, self.device)
        
        # 显示一共有多少参数要训练
        var_counts = tuple(core.count_vars(module) for module in [self.policy.pi])
        print('\n 训练的参数： \t pi: %d'%var_counts)

        # 创建记录数据的字典
        self.information = {}

    def compute_loss(self, data):
        '''
        计算q网络的loss
        '''
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        with torch.no_grad():
            temp_q = self.tar_policy.pi(o2).max(1)[0]       #  只是在这里把计算 temp q 的部分换成了目标网络！
            Q_targets = r + (1-d) * (self.gamma * temp_q)   # [batch_size, ]
            Q_targets = Q_targets.unsqueeze(1)              # [batch_size, 1]
        Q_expected_ = self.policy.pi(o)
        Q_expected = Q_expected_.gather(1, a.long())        # [batch_size, 1], a的形状必须是[batch_size, 1]

        loss = F.mse_loss(Q_expected, Q_targets)
        return loss
    
    
    def update(self, data):  # 这是和on-policy算法的区别之三，需要载入sample得到的data，on-policy在函数里面get
        '''
        更新步骤，--------------这是和on-policy算法的区别之三，需要载入sample得到的data，on-policy在函数里面get------------
        '''
        # 接下来对ac.pi进行优化
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # 记录
        self.information['LossPi'] = loss_pi.item()

        # 比DQN多了目标网络软更新！
        with torch.no_grad():
            for p, p_targ in zip(self.policy.parameters(), self.tar_policy.parameters()):
                p_targ.data.mul_(self.delay_up)
                p_targ.data.add_((1 - self.delay_up) * p.data)

    
    # 相比之前的 on-policy，还多了这两个方法！
    def get_action(self, o):
        '''
        只用于训练时产生轨迹和测试的时候选择动作
        其作用就是把np的obs转成tensor再使用step
        给1个状态，得到1个加噪声的动作，
         -----这是和on-policy算法的第四点区别，ddpg通过ac.act得到确定性动作后需要加噪声-----------
        '''
        a = self.policy.step(torch.as_tensor(o, dtype=torch.float32).to(self.device))  # get动作的时候也要把o变成device形状！
        return a
    
    def test_agent(self):
        '''定义一个测试智能体的函数，用来监控智能体的表现'''
        for j in range(self.num_test_epsodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # 采取不加噪声的动作
                o, r, d, _ = self.test_env.step(self.get_action(o))
                ep_ret += r
                ep_len += 1
            self.information['TestEpRet'] = ep_ret
            self.information['TestEpLen'] = ep_len
    









                



                
        
        


