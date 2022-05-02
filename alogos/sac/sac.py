# 必须要把根目录的路径添加进来！
import sys

from pytest import Instance

import alogos
sys.path.append('/home/zp/zpinup/')
from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import alogos.sac.core as core

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
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
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

class sac:
    '''
    这个只是连续版本的sac！离散的需要大改！因为这个是产生分布采集动作，所以不必像DDPg那样给动作加噪声！
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    '''
    def __init__(self,env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), 
                 replay_size=int(1e6), gamma=0.99, 
                 delayup=0.995, pi_lr=1e-3, q_lr=1e-3, alpha=0.2,
                 num_test_episodes=10, max_ep_len=1000,
                 device=None):
        self.device = device
        self.env, self.test_env = env_fn(), env_fn()
        self.obs_dim = self.env.observation_space.shape

        self.act_dim = self.env.action_space.shape[0]
        
        self.gamma = gamma
        self.alpha = alpha
        self.delay_up = delayup     # 延迟更新参数
        self.num_test_epsodes = num_test_episodes
        self.max_ep_len = max_ep_len
        # 建立ac model ac.pi, ac.q1, ac.q2
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(self.device)
        self.ac_targ = deepcopy(self.ac)   # 包括 ac_targ.pi 和 ac_targ.q1， ac_targ.q2
        
        # 技巧，把两个网络的参数合并到一起，itertools.cahin 
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.q_optimizer = Adam(self.q_params, lr=q_lr)                                      # q优化器同时优化 q1和q2的参数
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        
        # 初始buffer
        self.buffer = ReplayBuffer(self.obs_dim, self.act_dim, replay_size, self.device)

        # 冻结目标 ac_targ所有的 的参数, 包括q1, q2和pi，以减少计算资源
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # 显示一共有多少参数要训练
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        print('\n 训练的参数： \t pi: %d, \t q1: %d,  \t q2: %d\n'%var_counts)
        # 创建记录数据的字典
        self.information = {}
    
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done'] 
        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # 计算时序差分目标U
        with torch.no_grad():
            # 目标动作来自于现在的policy，这又叫做软状态值
            a2, logp_a2 = self.ac.pi(o2)   # a2也是重采样得到的
            # 计算时序差分目标u
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)
        
        # 计算时序差分误差
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # 一些有用的信息
        q_info = dict(Q1Vals=q1.cpu().detach().numpy().mean(), 
                      Q2Vals=q2.cpu().detach().numpy().mean())
        return loss_q, q_info

    def compute_loss_pi(self, data):
        o = data['obs']
        pi_act, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi_act)
        q2_pi = self.ac.q2(o, pi_act)
        q_pi = torch.min(q1_pi, q2_pi)

        # 带熵loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # 有用的信息
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy().mean())
        return loss_pi, pi_info
    
    def update(self, data):  # 这是和on-policy算法的区别之三，需要载入sample得到的data，on-policy在函数里面get。 这里是和ddpg的区别 多了一个计数器timer
        '''
        更新步骤，--------------这是和on-policy算法的区别之三，需要载入sample得到的data，on-policy在函数里面get------------
        '''
        # 先对ac.q网络进行1步优化
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # 记录
        self.information['LossQ'] = loss_q.item()
        self.information = Merge(self.information, q_info)

        # 冻结ac.q1, ac.q2网络，接下来更新策略pi的时候不要更改ac.q1, ac.q2
        for p in self.q_params:
            p.requires_grad = False
        
        # 接下来对ac.pi进行优化
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # 优化完 ac.pi，解冻ac.q
        for p in self.q_params:
            p.requires_grad = True
        
        # 记录
        self.information['LossPi'] = loss_pi.item()
        self.information = Merge(self.information, pi_info)


        # 最后， 更新target的3个网络参数，软更新 用了自乘操作，节省内存
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.delay_up)
                p_targ.data.add_((1 - self.delay_up) * p.data)
    
    # 相比之前的 on-policy，还多了这两个方法！
    def get_action(self, o, deterministic=False):
        '''
        只用于训练时候收集轨迹，和测试的时候选择动作
        给1个状态，得到1个动作
        '''
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.device), deterministic)  # 这里注意，o要转换为device。
    
    def test_agent(self):
        '''定义一个测试智能体的函数，用来监控智能体的表现'''
        for j in range(self.num_test_epsodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # 确定性动作
                o, r, d, _ = self.test_env.step(self.get_action(o, False))  # 不采用确定性动作
                ep_ret += r
                ep_len += 1
            self.information['TestEpRet'] = ep_ret
            self.information['TestEpLen'] = ep_len



        


