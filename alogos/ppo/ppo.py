# 必须要把根目录的路径添加进来！
import sys
sys.path.append('/home/zp/zpinup/')
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import alogos.ppo.core as core

class PPOBuffer:
    '''
    buffer 
    使用 generalized Advantage estimation（GAE-lambda）来估计优势函数 A(s,a)
    '''

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, device=None):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)  # observations
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)  # actions
        self.adv_buf = np.zeros(size, dtype=np.float32)                                # advantages
        self.rew_buf = np.zeros(size, dtype=np.float32)                                # rewards
        self.ret_buf = np.zeros(size, dtype=np.float32)                                # Gs
        self.val_buf = np.zeros(size, dtype=np.float32)                                # values
        self.logp_buf = np.zeros(size, dtype=np.float32)                               # log p(a)s
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, rew, val, logp):
        '''
        存入1个时间步的环境交互
        '''
        assert self.ptr < self.max_size     # buffer存取的限度要小于最大值
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        '''
        在一段轨迹结束时调用它，或者在一段轨迹因为done结束时调用它。
        last_val默认是0，但是可以手动输入它
        这会在缓冲区中回顾轨迹开始的位置，并使用整个轨迹中的奖励和价值估计来使用 GAE-Lambda 计算优势估计，并计算每个状态的奖励，以用作目标为价值函数。
        如果轨迹因智能体达到终端状态（死亡）而结束，则“last_val”参数应为 0，否则应为 V(s_T)，即为最后状态估计的值函数。
        这使我们能够引导进行奖励计算以考虑超出任意情节范围（或时期截止）的时间步长
        '''
        path_slice = slice(self.path_start_idx, self.ptr)  # slice返回切片索引的起始位置，比如说[start 0，end 10)
        rews = np.append(self.rew_buf[path_slice], last_val)  # 先得到切片data[0,1,...9]在发生终止的轨迹片段最后加上0值,最后轨迹的长度是11
        vals = np.append(self.val_buf[path_slice], last_val)

        # 使用GAR-lambda方法计算优势函数的估计值，注意切片是左开右闭区间[)，最后一位取不到
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]    # deltas的长度是10
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma*self.lam)

        # 计算G_t
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1] # 和deltas长度保持一致10，也就是最后的r=0舍弃了

        # 下一段轨迹的开始位置是这次轨迹的终止位置
        self.path_start_idx = self.ptr
    
    def get(self):
        '''
        在 1个epoch 结束的时候调用它，得到buffer中的所有数据，并且把优势归一化，并重置一些指针
        '''
        assert self.ptr == self.max_size  # 只有buffer满了才能get
        self.ptr, self.path_start_idx = 0, 0

        # 归一化优势advantages
        adv_mean, adv_std = core.get_mean_and_std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        # 把所有数据以字典的形式存起来
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)

        # 还是原来的字典，只不过数据变成tensor了
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k,v in data.items()}

class ppo:
    '''
    ppo类，定义了基本属性和更新过程
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    '''
    def __init__(self, env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), 
                steps_per_epoch=4000, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
                vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97,
                target_kl=0.01,  device=None
        ):
        '''
        PPO by clip
        '''
        self.device = device
        self.env = env_fn()
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape
        self.steps_per_epoch = steps_per_epoch
        self.clip_ratio = clip_ratio             # 这两个是ppo的重要超参数
        self.target_kl = target_kl
        self.train_v_iters = train_v_iters
        self.train_pi_iters = train_pi_iters     # 这个也是区别，pi也要多次训练
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(self.device)
        self.buf = PPOBuffer(self.obs_dim, self.act_dim, steps_per_epoch, gamma, lam, self.device)
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)

        # 计算一下一共要训练多少变量记录到log里，包括pi和v两个网络
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.v])
        print('\n 训练的参数： \t pi: %d, \t v: %d\n'%var_counts)
        # 创建记录数据的字典
        self.information = {}
                
    # 设置计算PPO的 policy loss, 不需要用到gamma，这个在buffer里面用过了。
    def compute_loss_pi(self, data):
        # 代码中ppo的old信息是直接从buffer读取的，而不是用一个旧的网络，因为参数更新完全基于自动梯度
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        # policy loss
        dist, logp = self.ac.pi(obs, act)   # 这里使用了pi的forward函数 ， [N, ] logp的形状
        # 这里是ppo的重点
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        # 有用的额外信息
        approx_kl = (logp_old - logp).mean().item()  # 返回元素值，近似kl散度，在trpo里面的kl是精准的kl散度
        ent = dist.entropy().mean().item()           # 熵
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)   # ratio比1+0.2 大或者ration比1-0.2 小， 返回的是 True Fasle这种bool量[True, Fasle...] [N,]
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()    # 返回平均值的纯数字
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info
    
    # 设置计算PPO的 value loss
    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs)- ret)**2).mean()
    
    def update(self):
        '''更新机制'''
        data = self.buf.get()
        # 在更新前得到pi和v的loss去掉梯度变成纯数值，并得到pi的info，相当于先备份
        pi_loss_old, pi_info_old = self.compute_loss_pi(data)
        pi_loss_old = pi_loss_old.item()    # item是得到纯数字
        v_loss_old = self.compute_loss_v(data).item()

        # 梯度下降法来更新pi，也更新好多次
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)    # 因为在这个小循环里面，更新一次pi之后，这一步算出来的值也会发生变化的，所以判断语句写下面了
            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                print('因为KL超过限定的KL，所以训练在%d 次更新终止'%i)
                break
            loss_pi.backward()
            self.pi_optimizer.step()

        self.information['StopIter'] = i

        # 更新tranv_iters次v
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)   # 在这 i 次里面，lossv会变化，因为v网络变了，所以算loss也变了
            loss_v.backward()
            self.vf_optimizer.step()
        
        # 记录更新前后损失和KL和熵的改变
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        self.information['LossPi'] = pi_loss_old
        self.information['LossV'] = v_loss_old
        self.information['KL'] = kl
        self.information['ClipFrac'] = cf
        self.information['Entropy'] = ent
        self.information['DeltaLossPi'] = (loss_pi.item() - pi_loss_old)
        self.information['DeltaLossV'] = (loss_v.item() - v_loss_old)
        

                 
