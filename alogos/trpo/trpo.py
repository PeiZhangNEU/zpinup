# 必须要把根目录的路径添加进来！
import sys
sys.path.append('/home/zp/zpinup/')
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import alogos.trpo.core as core
from copy import deepcopy
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

class TRPOBuffer:
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
    
class trpo:
    '''
    trpo类，定义了trpo的更新过程
    根据更新方法的不同，还包含了NPG这种算法的更新模式
    '''
    def __init__(self, env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), 
                steps_per_epoch=2048,
                gamma=0.99, lam=0.97, delta=0.01, 
                vf_lr=1e-3, train_v_iters=80, 
                backtrack_iter=10, backtrack_coeff=1.0, backtrack_alpha=0.5,
                device=None,
                mode=None
        ):
        self.device = device
        self.mode = mode   # 可以选择NPG或者TRPO这两种利用费社信息矩阵的算法
        self.env = env_fn()
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape
        self.steps_per_epoch = steps_per_epoch
        self.delta = delta
        self.backtrack_iter = backtrack_iter
        self.backtrack_coeff = backtrack_coeff
        self.backtrack_alpha = backtrack_alpha
        
        # 一个ac，和一个旧的actor
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(self.device)
        self.old_pi = deepcopy(self.ac.pi)

        # v 用正常的多次梯度下降来更新，但是pi用直线搜索
        self.train_v_iters = train_v_iters
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)
        
        self.buf = TRPOBuffer(self.obs_dim, self.act_dim, steps_per_epoch, gamma, lam, self.device)
        
        # 计算一下一共要训练多少变量记录到log里，包括pi和v两个网络
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.v])
        print('\n 训练的参数： \t pi: %d, \t v: %d\n'%var_counts)

        # 创建记录数据的字典
        self.information = {}
    
    def cg(self, obs, b, cg_iters=10, EPS=1e-8, residual_tol=1e-10):
         '''
         共轭梯度算法
         input: obs, \hat g_k(策略梯度的统计量)
         output: 直线搜索方向 (\alpha * \hat x_k)
                 其中 \hat x_k = \hat H_k ^(-1) * \hat g_k

         https://en.wikipedia.org/wiki/Conjugate_gradient_method
         传入obs是为了求kl， 传入的b是 \hat g_k 是策略梯度的估计值，
         输出是 直线搜索方向乘以一个\alpha  
         即 \alpha * \hat x_k = \alpha * (\hat H_k ^(-1) * \hat g_k)，
          因为求逆并不好求，所以下面的算法并没有求逆而是用了近似
         其中 \hat H_k 是对kl散度的海森矩阵，也就是kl对pi参数的二阶梯度矩阵
         '''
         x = torch.zeros(b.size()).to(self.device)
         r = b.clone()
         p = r.clone()
         rdotr = torch.dot(r,r).to(self.device)  # 一维向量对应位置相乘再求和，返回一个数值

         for _ in range(cg_iters):
             Ap = self.hessian_vector_product(obs, p)   
             alpha = rdotr / (torch.dot(p, Ap).to(self.device)+EPS)

             x += alpha * p
             r -= alpha * Ap

             new_rdotr = torch.dot(r,r)
             p = r + (new_rdotr / rdotr) * p
             rdotr = new_rdotr

             if rdotr < residual_tol:
                 break
         return x

    def hessian_vector_product(self, obs, p, damping_coeff=0.1):
        '''
        海森矩阵和统计量的乘积
        input: obs, \hat 统计量(任意常数向量)
        output: \hat H_k * \hat 统计量
        传入的 obs是为了计算kl， 然后求出kl对于pi的参数的海森矩阵拉直\hat H_k，再乘上一个常数统计量 \hat 统计量
        为什么里面有kl求的时候是新旧俩策略都是 self.ac.pi 因为就是要求它的二阶导，其中给一个的参数固定为常数
        '''
        p.detach()
        kl = self.compute_kl(new_policy=self.ac.pi, old_policy=self.ac.pi, obs=obs)   # 传入的都是ac.pi
        kl_grad = torch.autograd.grad(kl, self.ac.pi.parameters(), create_graph=True) # 创建graph为了二次求导
        kl_grad = self.flat_grad(kl_grad)

        kl_grad_p = (kl_grad * p).sum()
        kl_hessian = torch.autograd.grad(kl_grad_p, self.ac.pi.parameters())
        kl_hessian = self.flat_grad(kl_hessian, hessian=True)
        return kl_hessian + p * damping_coeff

    def compute_kl(self, new_policy, old_policy, obs):
        '''
        求解kl散度, 输入是ac.pi，产生distribution，然后直接用自带的kl函数求kl
        可以对高斯分布求kl，亦可以对其他分布求kl，如category的离散环境
        '''
        with torch.no_grad():   # 之前的dist里面所有的参数都不需要梯度，视为常数，因为求的是目前pi参数的海森矩阵
            dist_old = old_policy._distribution(obs)

        dist_new = new_policy._distribution(obs)
        kl_raw = kl_divergence(dist_new, dist_old)    # 这个函数可以计算gassian的kl，也可以计算category的kl
        if isinstance(dist_new, Normal):
            kl = kl_raw.sum(-1, keepdim=True).mean()  # 如果分布是gassian,kl_raw=[N, act_dim]， 
                                                      # 然后sum，keepdim保留第二维度变成[N, 1]，最后mean变成[]
        elif isinstance(dist_new, Categorical):
            kl = kl_raw.mean()                        # 如果分布是category，kl_raw=[N]
                                                      # 直接mean变成 []
        return kl
    
    def flat_grad(self, grads, hessian=False):
        '''
        把梯度展开成1行
        '''
        grad_flatten = []
        if hessian == False:
            for grad in grads:
                grad_flatten.append(grad.view(-1)) # 把矩阵展开成1行，比如(4,4)形状展开成(16)形状 
            grad_flatten = torch.cat(grad_flatten) # 把列表也横向合并，比如两个形状为[16],[16]组成的列表，最后得到[32]形状
            return grad_flatten
        elif hessian == True:
            for grad in grads:
                grad_flatten.append(grad.contiguous().view(-1)) # contigurous保证tensor展开后连续
            grad_flatten = torch.cat(grad_flatten).data
            return grad_flatten
    
    def flat_params(self, model):
        '''
        把模型的参数展开成1行
        '''
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        params_flatten = torch.cat(params)
        return params_flatten
    
    def update_model(self, model, new_params):
        '''
        用传入的new_params更新model.parameters()， new_params是1行的参数
        '''
        index = 0
        for params in model.parameters():
            params_length = len(params.view(-1))
            new_param = new_params[index : index + params_length]
            new_param = new_param.view(params.size())
            params.data.copy_(new_param)
            index += params_length
    

    # 设置计算TRPO的 policy loss即代理函数值, 不需要用到gamma，这个在buffer里面用过了。
    def compute_loss_pi(self, data):
        # old信息是直接从buffer读取的
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        # policy loss
        dist, logp = self.ac.pi(obs, act)   # 这里使用了pi的forward函数 ， [N, ] logp的形状

        # 这里是trpo的重点, 代理优势， 用代理优势函数近似策略期望函数
        # loss_pi = (log pa * adv ).mean() 这是最原始的策略目标函数，之前求策略梯度是对它求梯度
        ratio = torch.exp(logp - logp_old)
        loss_pi = (ratio * adv).mean()  # 注意！这里的loss没有负号了，因为我们是手动直线搜索,用的就是梯度上升
                                        # 之前ppo为了让一个目标函数max，是用adam等min优化器用的梯度下降，所以要加负号
                                        # 这个loss就是我们要 最大化的策略期望目标函数 的 近似函数值！
        return loss_pi
    
    # 设置计算TRPO的 value loss
    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs)- ret)**2).mean()
        
    def update(self):
        '''
        trpo的更新
        接受的都是批量数据
        '''
        data = self.buf.get()
        obs = data['obs']  # 需要用到一下obs求kl判断终止条件

        ########################################## 策略pi的更新 ###########################################
        # 计算policy的loss
        pi_loss_old = self.compute_loss_pi(data)

        # trpo重要步骤，求出代理函数正向梯度 \hat g_k，也就策略梯度的近似！
            # 需要求这个pi_loss_old（其实就是代理函数的值）的梯度
        gradient = torch.autograd.grad(pi_loss_old, self.ac.pi.parameters())
        gradient = self.flat_grad(gradient)
            # 使用共轭梯度算法计算出 alpha  * \hat x_k,也就是搜索方向
        search_dir = self.cg(obs, gradient.data)
            # x_kT H_k x_k
        gHg = (self.hessian_vector_product(obs, search_dir) * search_dir).sum(0)
            # 计算出直线搜索步骤中的根号式，也就是直线搜索的步长
        step_size = torch.sqrt(2 * self.delta / gHg)
        
        # self.old_pi 的两个作用：
        #   1. 保证目前old_policy和policy一致, old_pi的作用只是为了计算一下更新了pi之后的kl散度，来判断循环是否终止而已
        #   2. 另外一个是这个old_parms是作为一个备份，如果下面的直线搜索失败了，还能把参数还原给 ac.pi
        old_params = self.flat_params(self.ac.pi)
        self.update_model(self.old_pi, old_params)

        # 开始更新 pi
        # 选择更新模式，是NPG，还是TRPO
        if self.mode == 'NPG':
            # 普通的直线搜索, 就搜索一次
            params = old_params + step_size * search_dir  # 有方向和步长，直线搜索
            self.update_model(self.ac.pi, params)
            kl = self.compute_kl(new_policy=self.ac.pi, old_policy=self.old_pi, obs=obs)

        elif self.mode == 'TRPO':
            expected_improve = (gradient * step_size * search_dir).sum(0, keepdim=True)
            # backtracking 直线搜索，搜索若干次
            for i in range(self.backtrack_iter):
                params = old_params + self.backtrack_coeff * step_size * search_dir  # 多乘一个coeff
                self.update_model(self.ac.pi, params)     # 更新一次现在的ac.pi
                # 更新了一次 ac.pi 之后，再次用data计算现在的 loss_pi,注意ac.pi已经变化，但是logp_old还是来源于data，没变
                loss_pi = self.compute_loss_pi(data)

                loss_improve = loss_pi - pi_loss_old   # 计算现在的代理函数值和之前的差值
                
                expected_improve *= self.backtrack_coeff
                improve_condition = loss_improve / expected_improve
                self.information['ImproveCondition'] = improve_condition.item()

                kl = self.compute_kl(new_policy=self.ac.pi, old_policy=self.old_pi, obs=obs)

                if kl < self.delta and improve_condition > self.backtrack_alpha:
                    print('接受新的参数，在第 %d 步直线搜索'%i)
                    self.information['BackTrack_Iters'] = i
                    break

                if i == self.backtrack_iter-1:
                    print('直线搜索失败')
                    self.information['BackTrack_Iters'] = i
                    params = self.flat_params(self.old_pi)
                    self.update_model(self.ac.pi, params) # 如果直线搜索失败，就还原ac.pi的参数到更新开始前
                
                self.backtrack_coeff *= 0.5
                
        ########################################### 价值v的更新 ############################################
        # 更新tranv_iters次v
        v_loss_old = self.compute_loss_v(data)
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)   # 在这 i 次里面，lossv会变化，因为v网络变了，所以算loss也变了
            loss_v.backward()
            self.vf_optimizer.step()
        
        # 存储标量
        self.information['LossPi'] = pi_loss_old.item()
        self.information['LossV'] = v_loss_old.item()
        self.information['KL'] = kl.item()
                          



    










