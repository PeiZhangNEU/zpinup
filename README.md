基于spinningup的结构，但是不需要spinup的logx了

# 环境要求

需要安装mujoco210，自行研究

Package                         Version

attrs                           21.4.0

Box2D                           2.3.10

box2d-py                        2.3.8

glfw                            2.5.1

gym                             0.22.0

gym-notices                     0.0.6

gym-robotics                    0.1.0

mpi4py                          3.0.3

scipy                           1.7.3

seaborn                         0.11.2

tensorboard                     2.8.0

torch                           1.11.0

torchaudio                      0.11.0

torchvision                     0.12.0

tornado                         6.1

tqdm                            4.55.0

# 一、项目有哪些代码

策略梯度类： OpenAI

| 算法                                      | on/off policy | continous/discrete action |
| ----------------------------------------- | ------------- | ------------------------- |
| **vpg**                                   | on            | both                      |
| **trpo/npg**                              | on            | both                      |
| **ppo**                                   | on            | both                      |
| <font color='red'>ddpg(确定性动作)</font> | off           | con                       |
| <font color='red'>td3(确定性动作)</font>  | off           | con                       |
| sac                                       | off           | con                       |
| sac_discrete                              | off           | discrete                  |



值算法类：DeepMind

| 算法        | on/off policy | continous/discrete action |
| ----------- | ------------- | ------------------------- |
| Q_learning  | on            | discrete                  |
| **DQN**     | off           | discrete                  |
| **DQN2015** | off           | discrete                  |
| **DDQN**    | off           | discrete                  |
| DuelingDDQN | off           | discrete                  |

标黑色的三种算法在dqn文件夹里面集成



进阶：分层强化学习HRL

| 算法          | on/off policy | continous/discrete action |
| ------------- | ------------- | ------------------------- |
| Option_critic | off           | both                      |
| HDQN          | off           | discrete                  |
| ...待更新     |               |                           |



进阶：目标强化学习算法

| 算法     | on/off policy | continous/discrete action |
| -------- | ------------- | ------------------------- |
| HER_DDPG | off           | con                       |



# 二、代码格式

Spinning Up 项目的算法都按照固定的模板来实现。每个算法由两个文件组成：

- 算法文件，主要是算法的核心逻辑
- 核心文件，包括各种运行算法所需的工具类。
- train文件，训练
- test文件，测试训练好的网络

我都用Hopper这个环境，对比官网的效果。

**网络的forward函数，也就是net(xx)直接使用，一般都是在update函数里面，给批量的输入，用来批量更新！**

**而像 step, act, get_action 这种函数，一般都是给单个的输入，并且不需要grad，用于驱动环境运行，**



# 三、代码注解

## 3.1 策略梯度类

### On-Policy 回合更新(必须一个回合更新一次)

#### VPG

![image-20220309081653401](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220309081653401.png)

![image-20220304152547764](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220304152547764.png)

```python
# 设置计算VPG的 policy loss
    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        # policy loss
        dist, logp = self.ac.pi(obs, act)   # 这里使用了pi的forward函数
        loss_pi = -(logp * adv).mean()

        # 有用的额外信息
        approx_kl = (logp_old - logp).mean().item()  # 返回元素值
        ent = dist.entropy().mean().item()           # 熵
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info
    
    # 设置计算VPG的 value loss
    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs)- ret)**2).mean()

    
    def update(self):
        data = self.buf.get()
        # 在更新前得到pi和v的loss去掉梯度变成纯数值，并得到pi的info，相当于先备份
        pi_loss_old, pi_info_old = self.compute_loss_pi(data)
        pi_loss_old = pi_loss_old.item()
        v_loss_old = self.compute_loss_v(data).item()

        # 梯度下降法来更新pi,只更新一次
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # 每更新一次pi，更新tranv_iters次v
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            self.vf_optimizer.step()
        
        # 记录更新前后损失和KL和熵的改变
        kl, ent = pi_info['kl'], pi_info_old['ent']
        self.logger.store(LossPi=pi_loss_old, LossV=v_loss_old, 
                        KL=kl, Entropy=ent, 
                        DeltaLossPi=(loss_pi.item() - pi_loss_old), 
                        DeltaLossV=(loss_v.item() - v_loss_old))
```



VPG抄完了，发现训练效果还行，但是不是很好，可以直接用pytorch训练，在DRL环境，可根据自己的意图对代码进行重构。

spinup的一些小工具 utils可以用一用，但是也可以不用，用tensorboard绘图也挺好的。

离散表现：

![image-20220317110808939](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220317110808939.png)



![image-20220308204142465](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220308204142465.png)

![image-20220307150643026](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220307150643026.png)



#### TRPO 和 NPG

![image-20220309081253501](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220309081253501.png)

有别人写的代码，但是只能用于连续动作区间，理论上是可以用到离散动作空间的

用到了最优化课程里面的共轭梯度算法啊。

![image-20220307151512525](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220307151512525.png)



![image-20220307151354942](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220307151354942.png)

[Conjugate gradient method - Wikipedia](https://en.wikipedia.org/wiki/Conjugate_gradient_method)

代码需要用到共轭梯度算法，最优化学过的。

直线搜索用的不是一般的直线搜索，而是 `backtracking line search`

```python
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
                self.logger.store(ImproveCondition=improve_condition.item())

                kl = self.compute_kl(new_policy=self.ac.pi, old_policy=self.old_pi, obs=obs)

                if kl < self.delta and improve_condition > self.backtrack_alpha:
                    print('接受新的参数，在第 %d 步直线搜索'%i)
                    self.logger.store(BackTrack_Iters=i)
                    break

                if i == self.backtrack_iter-1:
                    print('直线搜索失败')
                    self.logger.store(BackTrack_Iters=i)
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
        self.logger.store(Losspi=pi_loss_old.item(), 
                          Lossv=v_loss_old.item(),
                          KL=kl.item(),
                          )
                        
```

离散表现

![image-20220317111947366](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220317111947366.png)

连续表现

![image-20220314160630386](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220314160630386.png)

![image-20220314160613015](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220314160613015.png)



```python
# self.old_pi 的两个作用：
        #   1. 保证目前old_policy和policy一致, old_pi的作用只是为了计算一下更新了pi之后的kl散度，来判断循环是否终止而已
        #   2. 另外一个是这个old_parms是作为一个备份，如果下面的直线搜索失败了，还能把参数还原给 ac.pi
        old_params = self.flat_params(self.ac.pi)
        self.update_model(self.old_pi, old_params)
```

NPG只不过是update和trpo不同，只用了一次直线搜索，其他都一样！



#### PPO

![image-20220309081745829](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220309081745829.png)

![image-20220309081843053](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220309081843053.png)

虽然算法中出现了 $\theta_k, \theta_{k+1}$ 但是，更新参数的时候是靠loss的纯梯度自动更新，所以只需要1个policy就行，不需要再搞一个旧的policy。旧的量就用buffer中存储的就可以了。因为buffer中存储的是上一个epoch的策略产生的值，这一个epoch的策略已经是更新过的策略了。

![image-20220308192451763](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220308192451763.png)



离散表现

![image-20220317112232111](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220317112232111.png)

![image-20220309082203491](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220309082203491.png)

<img src="https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220308204313906.png" alt="image-20220308204313906" style="zoom:67%;" />

```python
# 设置计算VPG的 policy loss
    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        # policy loss
        dist, logp = self.ac.pi(obs, act)   # 这里使用了pi的forward函数 ， [N, ] logp的形状
        # 这里是ppo的重点
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        # 有用的额外信息
        approx_kl = (logp_old - logp).mean().item()  # 返回元素值
        ent = dist.entropy().mean().item()           # 熵
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)   # ratio比1+0.2 大或者ration比1-0.2 小， 返回的是 True Fasle这种bool量[True, Fasle...] [N,]
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()    # 返回平均值的纯数字
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info
    
    # 设置计算VPG的 value loss
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
            loss_pi, pi_info = self.compute_loss_pi(data)          # 因为在这个小循环里面，更新一次pi之后，这一步算出来的值也会发生变化的，所以判断语句写下面了
            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                self.logger.log('因为KL超过限定的KL，所以训练在%d 次更新终止'%i)
                break
            loss_pi.backward()
            self.pi_optimizer.step()

        self.logger.store(StopIter=i)  # 记录下来这因为kl大了终止的更新的次数i

        # 更新tranv_iters次v
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)   # 在这 i 次里面，lossv会变化，因为v网络变了，所以算loss也变了
            loss_v.backward()
            self.vf_optimizer.step()
        
        # 记录更新前后损失和KL和熵的改变
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        self.logger.store(LossPi=pi_loss_old, LossV=v_loss_old, 
                            KL=kl, Entropy=ent, ClipFrac=cf,
                            DeltaLossPi=(loss_pi.item() - pi_loss_old), 
                            DeltaLossV=(loss_v.item() - v_loss_old))
```





#### 总结 `self.ac.step(obs)`和`self.ac.pi.forward(obs, act)`

两个标志性的函数

在 core 里面 的 actorcritic 类，有一个`step`函数：**策略根据obs<font color='red'>自己产生动作a</font>，然后求这个动作的概率**，为了传到buff里面，所以不需要梯度

```python
def step(self, obs):
        '''
        专门用来用pi产生动作并求概率，产生的old_log_pa
        当然N可以为1
        给[N,obs_dim]的一批状态,它和forward的区别就在于没有梯度，并且输入只需要obs
        不用梯度，测试的输出该状态下
        使用策略得到的动作， 状态的价值， 动作对应的log p(a)
        '''
        with torch.no_grad():
            dist = self.pi._distribution(obs)
            a = dist.sample()
            logp_a = self.pi._log_prob_from_distribution(dist, a)

            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()
```

它是不需要梯度的，传入1个obs，输出一个动作以及这个动作对应的log, 用来驱动env运行

这个函数是专门为了产生 `old_logp = old_pi(a|s)` 的。是用来往buf里面传的



在core里面的Actor类有个 forward函数，也就是 actorcritic.actor 里面的函数。 **策略根据 obs 和 <font color='red'>传入的 a</font> ，求出对应的概率**

```python
def forward(self, obs, act=None):
        '''
        只在upadate这一步计算loss时才需要用到
        带梯度
        产生给定状态的分布dist
        计算分布下，给定动作对应的log p(a)
        actor里面forward一般是只接收批量的数据，每一步的计算用上面的函数
        '''
        dist = self._distribution(obs)   # \pi(\cdot|s)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(dist, act)
        return dist, logp_a
```

它是需要梯度的，它是传入批次的 obs 和 act，用新的策略计算 `new_logp = new_pi(a|s)` 的，返回dist和logpa







### Off-Policy 时序差分更新(可以单步也可以回合更新)

先收集数据，如果env终止了，再重启继续收集。

如果到时间更新网络了，那就更新网络

只有一层循环！每次update需要从经验池抽取数据

```python
for t in range(total_steps):
    ...
    if t > update_start_step and t % freq==0:
        update(get data form expbuf)
```

并且在update更新**价值（v或者q）**的时候，用到的是时序差分，无论是q还是v，都是使用的$y$ 和q或者v的MSE，即 $U = y = r+\gamma v(或者q)$

![image-20220315132112938](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220315132112938.png)

![image-20220315132124571](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220315132124571.png)

写程序的时候一定要注意，是梯度上升还是下降！

如果是梯度下降，那么loss就是公式本身

如果是梯度上升，那么loss是取负

#### DDPG

![image-20220309090240190](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220309090240190.png)

![image-20220311185006713](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220311185006713.png)

```python
def compute_loss_q(self, data):
        '''
        计算q网络的loss
        '''
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q = self.ac.q(o, a)
        # 计算 y(r,s',d) ，也就是Q目标估计
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))  # s', a'=ac_tar.pi(s')
            backup = r + self.gamma * (1 - d) * q_pi_targ
        # MSE loss 
        loss_q = ((q - backup)**2).mean()
        # 有用的info
        loss_info = dict(QVals=q.cpu().detach().numpy())   # 形状 [N,]
        return loss_q, loss_info
    
    def compute_loss_pi(self, data):
        '''
        计算确定性策略的策略loss
        '''
        o = data['obs']
        q_pi = self.ac.q(o, self.ac.pi(o))  # s, a = ac.pi(s)
        return -q_pi.mean()
    
    def update(self, data):  # 这是和on-policy算法的区别之三，需要载入sample得到的data，on-policy在函数里面get
        '''
        更新步骤，--------------这是和on-policy算法的区别之三，需要载入sample得到的data，on-policy在函数里面get------------
        '''
        # 先对ac.q网络进行1步优化
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # 冻结ac.q网络，接下来更新策略pi的时候不要更改ac.q
        for p in self.ac.q.parameters():
            p.requires_grad = False
        
        # 接下来对ac.pi进行优化
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # 优化完 ac.pi，解冻ac.q
        for p in self.ac.q.parameters():
            p.requires_grad = True
        
        # 记录
        self.logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # 最后， 更新target的两个网络参数，软更新 用了自乘操作，节省内存
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.delay_up)
                p_targ.data.add_((1 - self.delay_up) * p.data)
```

![image-20220317130210676](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220317130210676.png)

![image-20220314203828354](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220314203828354.png)



#### TD3

两个ac，一个ac3个网络，pi+q1+q2

![image-20220314191013293](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220314191013293.png)

![image-20220317105608586](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220317105608586.png)

更新步骤和ddpg一样，只不过计算损失不一样了，并且ac多了个q2网络

```python
 def compute_loss_q(self, data):
        '''
        计算q网络的loss， 比之前多了个q网络而已。
        '''
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)
        
        
        with torch.no_grad():
            # 计算动作 a' 也就是 pi 目标估计
            pi_targ = self.ac_targ.pi(o2)
            # 目标策略平滑处理
            epsilon = torch.randn_like(pi_targ) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -self.act_limit, self.act_limit)

            # 计算 y(r,s',d) ，也就是Q目标估计
            q1_pi_targ = self.ac_targ.q1(o2, a2)  # s', a'=ac_tar.pi(s') + epsilon
            q2_pi_targ = self.ac_targ.q2(o2, a2) 
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss 
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # 有用的info
        loss_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                         Q2Vals=q2.cpu().detach().numpy())   # 形状 [N,]
        return loss_q, loss_info
    
    def compute_loss_pi(self, data):
        '''
        计算确定性策略的策略loss
        '''
        o = data['obs']
        q1_pi = self.ac.q1(o, self.ac.pi(o))  # s, a = ac.pi(s)
        return -q1_pi.mean()
    
    def update(self, data, timer):  # 这是和on-policy算法的区别之三，需要载入sample得到的data，on-policy在函数里面get。 这里是和ddpg的区别 多了一个计数器timer
        '''
        更新步骤，--------------这是和on-policy算法的区别之三，需要载入sample得到的data，on-policy在函数里面get------------
        '''
        # 先对ac.q网络进行1步优化
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # 记录
        self.logger.store(LossQ=loss_q.item(), **loss_info)

        # 这里是和ddpg的重要区别， 延迟更新pi
        if timer % self.policy_delay == 0:

            # 冻结ac.q1, ac.q2网络，接下来更新策略pi的时候不要更改ac.q1, ac.q2
            for p in self.q_params:
                p.requires_grad = False
            
            # 接下来对ac.pi进行优化
            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optimizer.step()

            # 优化完 ac.pi，解冻ac.q
            for p in self.q_params:
                p.requires_grad = True
            
            # 记录
            self.logger.store(LossPi=loss_pi.item())

            # 最后， 更新target的3个网络参数，软更新 用了自乘操作，节省内存
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    p_targ.data.mul_(self.delay_up)
                    p_targ.data.add_((1 - self.delay_up) * p.data)
```

![image-20220317125534090](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220317125534090.png)

![image-20220314203857340](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220314203857340.png)

#### SAC (2018) 不带温度参数

原文2018SAC，是由一个策略，2个v，2个q组成的

但是 spinningup把 Bellman方差 把 原来的 q用 v表示变成了 q用 q'表示，所以SAC是

由1个策略，4个q网络组成！q包括两个，以及两个目标网络

SAC2018中，熵参数 $\alpha$ 是固定的。到了SAC2019，这个熵参数 $\alpha$ 也变成一个可训练量！



SAC使用的是无限视野

更新价值

SAC的两个特点：

1. **压扁**！同样是使用高斯分布来sample动作，但是SAC 采集出动作之后，先使用了 tanh 把动作变到-1 , 1 之间，然后再乘上limit映射到动作的范围中！这一点是之前PPO，TRPO，VPG所不具备的。

   这一步变化，导致了下面的式子

   ![image-20220316092824604](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220316092824604.png)

2. 同样是使用高斯分布，但是SAC用神经网络输出的是**两组数** $\mu,\sigma$，他把这两个都作为了网络的输出。而PPO，TRPO，VPG这几种也使用分布的算法，只是用神经网络输出 $\mu$，而把方差作为一个单独的变量进行优化。（**为什么在SAC中使用单独方差会失灵？因为使用了重参数化，重参数化时需要用到网络输出的方差进行重参数化，如果把方差作为单独变量，会导致重参数化之后的动作与之前的网络梯度中断！**）

3. 使用了**重参数化技巧**，因为优化的时候需要求一个 Q网络对于动作的一阶导数再求对$\theta$ 的二阶导数，所以动作需要**进行重参数化手段才能有二阶梯度**，否则只用平时的采样会没有梯度！

   ![image-20220316095250413](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220316095250413.png)

![image-20220316094411131](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220316094411131.png)

![image-20220315084924720](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220315084924720.png)

![image-20220315132757757](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220315132757757.png)

![image-20220315132815216](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220315132815216.png)

更新策略

![image-20220315143025988](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220315143025988.png)

![image-20220315143037726](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220315143037726.png)

![image-20220317091226705](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220317091226705.png)

通过伪代码，我们可以看到，

关于 $\pi_\theta(a|s)$ 都是只需要计算 
$$
\pi(\tilde a' |s')
$$
其中 $\tilde a' \sim \pi(.|s')$ 是直接输入s'给策略，然后策略产生分布 dist， 然后rsample（重参数化）出来的。

不需要我们使用从buffer 中采集动作 a，然后计算 $\pi(a|s)$。

所以，actor函数的forward不需要有动作传入，也就是说，不需要 `forward(s, a)`。actor网络仅仅接收 s 即可！

也就是说，现在的actor的主要功能`forward`仅需要和之前的 `ac.step` 这个函数一样，**仅需要自己产生动作然后求概率，不需要接收外部的动作求概率**！

`forward(s)` 函数需要既支持批量传入，也需要支持单个传入，驱动环境运行！求动作必须要带梯度！因为![image-20220315170614775](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220315170614775.png)函数需要从分布里面采集动作，需要梯度！

**SAC的forward包含了之前`ac.step`函数的功能，并且不需要传入其他动作求概率，所以SAC程序里面没有`ac.step`函数了**



Normal分布的sample和rsample的区别，**rsample是带梯度的sample，全称叫做 reparametrization trick**！在forward函数里面，必须要用rsample！



重参数化的公式是：

![image-20220316092554240](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220316092554240.png)

但是pytorch的dist自带的`dist.rsample()`函数的公式是

![image-20220316101420106](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220316101420106.png)

**我们需要在pytorch的rsample之后，手动加一个tanh！**这样才能真正达到重参数化！



使用了 重参数化之后，使用tan进行压扁！



$u$=![image-20220316092554240](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220316092554240.png)

我们想要求原来没有变形的 $a$ 的 $log\pi(a|s)$ 就需要用下面的式子来计算！ 

![image-20220316092824604](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220316092824604.png)

![image-20220317083709866](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220317083709866.png)

![image-20220315164145655](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220315164145655.png)

```python
	def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done'] 
        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # 计算时序差分目标U
        with torch.no_grad():
            # 目标动作来自于现在的policy
            a2, logp_a2 = self.ac.pi(o2)
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
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(), 
                      Q2Vals=q2.cpu().detach().numpy())
        return loss_q, q_info

    def compute_loss_pi(self, data):
        o = data['obs']
        pi_act, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi_act)
        q2_pi = self.ac.q2(o, pi_act)
        q_pi = torch.min(q1_pi, q2_pi)

        # 带熵loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()  # 因为是梯度上升，所以和伪代码是相反数

        # 有用的信息
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())
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
        self.logger.store(LossQ=loss_q.item(), **q_info)

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
        self.logger.store(LossPi=loss_pi.item(), **pi_info)

        # 最后， 更新target的3个网络参数，软更新 用了自乘操作，节省内存
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.delay_up)
                p_targ.data.add_((1 - self.delay_up) * p.data)
```



![image-20220317125702016](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220317125702016.png)

![image-20220315191441506](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220315191441506.png)





#### SAC_Discrete (2019) 带温度参数

SAC_Discrete 是在 SAC2019的基础上进行改造的，具体参考

[1910.07207.pdf (arxiv.org)](https://arxiv.org/pdf/1910.07207.pdf) 离散SAC

SAC2019 和 2018 最大的区别就是，多了一个 自动优化的熵参数 $\alpha$ ，把这个参数也作为训练目标

也就是引入了一个温度参数的代价函数去优化 $\alpha$ ，其它的优化函数和之前一致。

![image-20220316115657877](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220316115657877.png)



如何改SAC为离散！

[深度强化学习-为离散动作空间调整Soft Actor Critic - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/438731114)

[Felhof/DiscreteSAC (github.com)](https://github.com/Felhof/DiscreteSAC)

离散的SAC

主要就是把q函数的结构给改了

**原来SAC：**

**q(s_dim+a_dim) ——> 1**

**现在SAC**

**q(s_dim) ——> a_dim**

其主要思想其实是参考了DQN这个很原始的专门处理离散问题的网络！

具体的改进过程见我的ipad！



**这个程序在主循环中，不可以一次update太多次，比如cartPole这个环境，很容易就达到每次奖励都是200整了，那么就会导致后续求梯度的时候，梯度消失。**

想想看，如果每次奖励都一样都是200，最后的梯度会是0！



```python
def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done'] 
        q1 = self.ac.q1(o)  # [N, n]
        q2 = self.ac.q2(o)  # [N, n]
        

        # 计算时序差分目标U
        with torch.no_grad():
            # 目标动作来自于现在的policy，这又叫做软状态值
            _, pi2, logp_pi2 = self.ac.pi(o2)   #  pi(o2) [N,n] logp_pi(o2) [N,n]
            # 计算时序差分目标u
            q1_pi_targ = self.ac_targ.q1(o2)  # [N, n]
            q2_pi_targ = self.ac_targ.q2(o2)  # [N, n]
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)  # [N, n]
            soft_state_values = (pi2 * (q_pi_targ - self.alpha * logp_pi2)).sum(dim=1)  #[N, ]
            backup = r + self.gamma * (1 - d) * (soft_state_values)   # [N, ]

        # gather函数要好好看看怎么用的！
        soft_q1 = q1.gather(1, a).squeeze(-1)   # [N, ]
        soft_q2 = q2.gather(1, a).squeeze(-1)   # [N, ]
        # 计算时序差分误差
        loss_q1 = ((soft_q1 - backup)**2).mean()
        loss_q2 = ((soft_q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # 一些有用的信息
        q_info = dict(Q1Vals=soft_q1.cpu().detach().numpy(), 
                      Q2Vals=soft_q2.cpu().detach().numpy())
        return loss_q, q_info

    def compute_loss_pi(self, data):
        o = data['obs']
        _, pi, logp_pi = self.ac.pi(o)   #  pi(o) [N,n] logp_pi(o) [N,n]
        q1_pi = self.ac.q1(o)            #  q1_pi [N, n]
        q2_pi = self.ac.q2(o)            #  q2_pi [N, n]
        q_pi = torch.min(q1_pi, q2_pi)   #  q_pi  [N, n] 

        # 带熵loss
        inside_term = self.alpha * logp_pi - q_pi          # 一定要注意，这里是梯度上升，原算法里面写的优化公式要取负值。
        loss_pi = (pi * inside_term).sum(dim=1).mean()
        
        # 有用的信息
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())
        return loss_pi, logp_pi, pi_info

    def compute_alpha_loss(self, logp_pi):
        '''利用loss pi 返回的logp_pi 来计算loss alpha'''
        loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean() 
        return loss_alpha
    
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
        self.logger.store(LossQ=loss_q.item(), **q_info)

        # 冻结ac.q1, ac.q2网络，接下来更新策略pi的时候不要更改ac.q1, ac.q2
        for p in self.q_params:
            p.requires_grad = False

        # 接下来对ac.pi进行优化
        self.pi_optimizer.zero_grad()
        loss_pi, log_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # 记录
        self.logger.store(LossPi=loss_pi.item(), **pi_info)

        # 优化完 ac.pi，解冻ac.q
        for p in self.q_params:
            p.requires_grad = True

        # 接下来对温度进行优化
        self.alpha_optimizer.zero_grad()
        loss_alpha = self.compute_alpha_loss(log_pi)
        loss_alpha.backward()
        self.alpha_optimizer.step()

        # 前面先用没有梯度的alpha进行计算，这里再对这个alpha进行更新。
        self.alpha = self.log_alpha.exp()

        # 记录
        self.logger.store(Alpha=self.alpha.item())
        
        

        # 最后， 更新target的3个网络参数，软更新 用了自乘操作，节省内存
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.delay_up)
                p_targ.data.add_((1 - self.delay_up) * p.data)
```



主要改变的网络就是变成 Category的分布！

```python
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
            logp_pi = pi_distribution.logits  # ln(pi(s))  【N，act_dim】
            
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
```





![image-20220317125820023](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220317125820023.png)





## 3.2 分层强化学习



### 基于Option的分层强化学习

在基于option的分层强化学习中，上层控制器根据上层策略选择options，下层控制器根据所选择的option所对应的策略选择action，从而实现分层。

#### Option-Critic: The Option-Critic Architecture 

[Bacon et al_2016_The Option-Critic Architecture.pdf](E:\zotero_moren\allresearchs\分层强化学习\Bacon et al_2016_The Option-Critic Architecture.pdf) 

[【分层强化学习】The Option-Critic Architecture 阅读笔记 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/430471198)

**Option自己的推导： [option critic.pdf](option critic.pdf)** 

首先是增强状态的定义

$(s,\omega)$，$\omega\in\Omega$

$\omega$ 是一个 option，它是一个三元组 $(I_\omega, \pi_\omega,\beta_\omega)$

强化学习整个网络的结构如下

针对 lunarLander 任务，**动作维度为4**，**状态维度为 8**, **optin_num 为 2**，即$\omega=0\ or\ 1$

公共部分 features            把 observation 转换为features

$Q_{\Omega}$：Q_layer                    给出两个option对应的状态价值，**类比 V(s)!**

$\beta_{\Omega}$：termination_layer   给出两个option终止的概率

$\pi_\omega$：option_layer   内策略，两个option对应两个策略

option: 就是$\omega$ 就是 0 或者 1，因为option_num 是2

```
opt_cri_arch(
  (feature): Sequential(
    (0): Linear(in_features=8, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=128, bias=True)
    (3): ReLU()
  )
  (q_value_layer): Linear(in_features=128, out_features=2, bias=True)
  (termination_layer): Linear(in_features=128, out_features=2, bias=True)
  (option_layer): ModuleList(
    (0): Linear(in_features=128, out_features=4, bias=True)
    (1): Linear(in_features=128, out_features=4, bias=True)
  )
)
```



#### env.unwrapped

小知识！

[gym中env的unwrapped_星之所望的博客-CSDN博客](https://blog.csdn.net/weixin_42769131/article/details/114550177)

![image-20220510153810092](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220510153810092.png)

对于不会返回done的程序，不要使用unwrapped！

比如 pendulum，就不能用 unwrapped， 因为本身done就一直等于 False！ 如果再不加次数限制，会死循环！



改出来了一份连续动作空间的 option-critic! 更新的batchsize 不要太大！ 64就行， 小批次下降会带来更好的效果! 

一份pendulum好用的超参数

```python
 test = option_critic(
        env=env,
        episode=250,
        exploration=2000,
        update_freq=4,
        freeze_interval=200,
        batch_size=64,
        capacity=100000,
        learning_rate=3e-4,
        option_num=4,
        gamma=0.99,
        termination_reg=0.01,
        epsilon_init=1.,
        decay=10000,
        epsilon_min=0.01,
        entropy_weight=1e-2,
        conv=False,
        cuda=cuda,
        render=True,
        save_path='./model/pendulum.pkl'
    )
    test.run()
```



### 基于Goal的分层强化学习

在基于goal的分层强化学习中，上层控制器在一个较长的时间跨度上根据上层策略选择一个goal，而下层控制器则在一个较短的时间跨度上根据所选择的goal以及下层策略选择action，目标是实现goal。因此，一个十分关键的问题是——**如何定义goal**？在上一节所介绍的UVFA与HER算法中，目标空间即为状态空间，目标就是达到某一特定的状态。实际上对于目标，不同的论文有不同的定义方法，因此在介绍每篇论文时，我将首先明确论文中goal的具体含义，然后再简单介绍论文工作。



#### H-DQN

[【分层强化学习】H-DQN：Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction阅读笔记 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/430886508)





## 3.3 DQN族

zpinup中

DQN2013， DQN2015 以及 DDQN 都放在了dqn代码里， 因为他们的core都一样，包括网络的结构和get_action的方法都一样。

DuelingDDQN 单独存放了



和AC不同的是，**Q算法选择动作和计算价值都是通过Q价值网络的，根据值来选择动作，贪婪策略**

注意下，离散动作空间对应的 动作价值网络 $Q(s,a)$ 的输入不是s和a的cat然后输出1个价值，而是输入s，**计算 Q(s),  输出a个价值**！

但是对于状态价值  $V(s)$ , 无论是离散还是连续动作空间， 网络都是输入s，输出维度1的价值，和动作无关！

![image-20220513100028019](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220513100028019.png)



### Q-Learning

原本来说，只能处理

**observation是离散的**

**action也是离散的环境**

如果类似 lunarlander 等 obs是连续的box，而action 是离散的discrete n的环境，需要先把obs空间离散化！

具体我代码写的有。

![image-20220513085743355](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220513085743355.png)

### DQN

#### 2013初版：只有1个Q网络

**Q 网络 ，就是动作价值函数， Q算法选择动作和计算价值都是通过Q价值网络的，根据值来选择动作，贪婪策略**

if random > epsilon:

​	a = argmax(Q(s))

else:

​	a = randm

 [Mnih et al_Playing Atari with Deep Reinforcement Learning.pdf](E:\zotero_moren\allresearchs\强化学习基础论文\Mnih et al_Playing Atari with Deep Reinforcement Learning.pdf) 

![image-20220513090431227](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220513090431227.png)

```python
def compute_loss(self, data):
        '''
        计算q网络的loss
        '''
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        with torch.no_grad():
            temp_q = self.policy.pi(o2).max(1)[0] 
            Q_targets = r + (1-d) * (self.gamma * temp_q)   # [batch_size, ]
            Q_targets = Q_targets.unsqueeze(1)              # [batch_size, 1]
        Q_expected_ = self.policy.pi(o)
        Q_expected = Q_expected_.gather(1, a.long())        # [batch_size, 1], a的形状必须是[batch_size, 1]

        loss = F.mse_loss(Q_expected, Q_targets)
        return loss
```





#### 2015改进版： 2个Q网络，目标网络缓慢更新

 [Mnih et al_2015_Human-level control through deep reinforcement learning.pdf](E:\zotero_moren\allresearchs\强化学习基础论文\Mnih et al_2015_Human-level control through deep reinforcement learning.pdf) 

![image-20220513090529018](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220513090529018.png)

```python
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
```



### DDQN(double DQN)

把**2015 DQN** 的y换成下式。就这个y变了，其他都不变。

 [van Hasselt et al_Deep Reinforcement Learning with Double Q-learning.pdf](E:\zotero_moren\allresearchs\强化学习基础论文\van Hasselt et al_Deep Reinforcement Learning with Double Q-learning.pdf) 

![image-20220513090546292](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220513090546292.png)

仔细对比一下，发现主要是里面这部分不一样了！本来是求整体max，DDQN是把最大值a直接计算出来放进去。

![image-20220513090747127](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220513090747127.png)

![image-20220513090658374](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220513090658374.png)



```python
def compute_loss(self, data):
        '''
        计算q网络的loss
        '''
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        with torch.no_grad():

            #  和 dqn 2015不同的就是和这里，计算temp_q 不一样！
            max_action_indexes = self.policy.pi(o2).argmax(1)     # [batch_size, ]， 求q表里面最大值对应的动作
            max_action_indexes = max_action_indexes.unsqueeze(1)  # [batch_size, 1]
            temp_q = self.tar_policy.pi(o2).gather(1, max_action_indexes)  # [batch_size, 1]
                                            # gather 里面后面的索引必须也是和tensor形状的维度一致的tensor  [batch_size, 1]
            temp_q = temp_q.squeeze()       # 变成 [bathc_size, ] 因为下面要计算乘积

            Q_targets = r + (1-d) * (self.gamma * temp_q)   # [batch_size, ]
            Q_targets = Q_targets.unsqueeze(1)              # [batch_size, 1]
        Q_expected_ = self.policy.pi(o)
        Q_expected = Q_expected_.gather(1, a.long())        # [batch_size, 1], a的形状必须是[batch_size, 1]

        loss = F.mse_loss(Q_expected, Q_targets)
        return loss
```



### DuelingDDQN

论文直接在 **DDQN 基础**上进行的改进

 [Wang et al_2016_Dueling Network Architectures for Deep Reinforcement Learning.pdf](E:\zotero_moren\allresearchs\强化学习基础论文\Wang et al_2016_Dueling Network Architectures for Deep Reinforcement Learning.pdf) 

[Torch | Dueling Deep Q-Networks](http://torch.ch/blog/2016/04/30/dueling_dqn.html)

torch建议的改进！

![image-20220513103303405](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220513103303405.png)

![image-20220513104653165](https://github.com/PeiZhangNEU/zpinup/blob/master/zpinupgit_assets/image-20220513104653165.png)



实际代码怎么写？

可以输入s ，网络的输出维度为 a_dim+1 , 最后一个神经元输出的是 $v(s)$ ，而前面的a_dim个代表 $Q(s)[a]$ 

进行输出动作时，把网络的输出的 Advantage单独提取出来，进行argmax即可。

```python
def step(self, obs):
    '''
    只接受1个obs，用于驱动环境运行
    贪婪策略
    在训练时，动作用这个产生, 这个过程本身就是贪婪策略了，在train的时候直接用即可
    '''
    if np.random.uniform() >= self.epsilon:
        with torch.no_grad():
            logits = self.pi(obs).cpu().numpy()
            logits = logits.squeeze()             # 为什么要squeeze 因为 atari的时候 logtis形状是 [1, act_dim+1] 前面多了个1
            # logits 输出的维度本来是 act_dim + 1 选择动作的时候根据前act_dim的 A(s)[a]选
            logits_action = logits[:-1]
            a = np.argmax(logits_action)
    else:
        a = np.random.randint(0, self.act_dim)
```

计算损失的函数如下：

```python
def calculate_duelling_q_values(self, duelling_network_output):
        """
        在计算loss的时候使用的内函数
        利用dueling net的结构，计算出来 A 和 V
        然后估计 Q 按照论文中的估计公式来估计
        """
        # 把 V(s) 拆出来
        state_value = duelling_network_output[:, -1]          # [batch_size, ]
        # 把 A(s)[a=.] 拆出来
        advantage_value = duelling_network_output[:, :-1]   # [batch_size, act_dim]
        # 计算mean A
        avg_advantage = torch.mean(advantage_value, dim=1)    # [batch_size, ]

        # 计算 估计的Q
        q_values = state_value.unsqueeze(1) + (advantage_value - avg_advantage.unsqueeze(1))  # [batch_size, act_dim]
        return q_values

    def compute_loss(self, data):
        '''
        计算q网络的loss
        '''
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        with torch.no_grad():

            #  和 ddqn 不同的就是和这里，计算temp_q 不一样！
            # 求最大动作索引的时候是按照 A(s)[a]求的，所以要把网络的act_dim+1 的输出保留前几维度
            max_action_indexes = self.policy.pi(o2)[:, :-1].argmax(1)    # [batch_size, ]， 求q表里面最大值对应的动作

            max_action_indexes = max_action_indexes.unsqueeze(1)     # [batch_size, 1]
                # 计算 temp q 需要多一步骤
            dueling_tar_net_outputs = self.tar_policy.pi(o2)         # [batchsize, act_dim+1]
            dueling_tar_q = self.calculate_duelling_q_values(dueling_tar_net_outputs)  # [batch_size, act_dim]  取代了之前ddqn的self.tar_policy.pi(o2)

            temp_q = dueling_tar_q.gather(1, max_action_indexes)  # [batch_size, 1]
                                            # gather 里面后面的索引必须也是和tensor形状的维度一致的tensor  [batch_size, 1]
            temp_q = temp_q.squeeze()       # 变成 [bathc_size, ] 因为下面要计算乘积

            Q_targets = r + (1-d) * (self.gamma * temp_q)   # [batch_size, ]
            Q_targets = Q_targets.unsqueeze(1)              # [batch_size, 1]
        
        # 和 ddqn 不一样的是，计算 现在的Q 也不一样！
        dueling_net_outputs = self.policy.pi(o)                                # [batchsize, act_dim+1]
        Q_expected_ = self.calculate_duelling_q_values(dueling_net_outputs) # [batch_size, act_dim]  取代了之前ddqn的self.policy.pi(o)
        Q_expected = Q_expected_.gather(1, a.long())                        # [batch_size, 1], a的形状必须是[batch_size, 1]

        loss = F.mse_loss(Q_expected, Q_targets)
        return loss
```





### Atari 支持

在DQN大类的core函数放入了可以训练Atari的模块：

```python
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
```

创建 agent.policy.pi的时候



#### DQN大类： DQN DDQN

根据 conv 标志，确定网络的结构， 以及forward的计算方法

```python
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
```



#### DuelingDDQN

它的Q net 的输出维度是 act_dim +1 和上面有所不同

```python
# Actor的基础类以及离散Actor和连续Actor类。
class Q_net(nn.Module):
    '''
    创建actor
    类里面的方法出现前置下划线是，代表这个函数是该类私有的，只能在内部调用
    这个类没有 __init__(self, inputs) 所以是不可实例化的类，只是一个用来继承的模板
    '''
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, useconve=False):
        '''初始一个logits网络，可以直接输出各个动作对应的概率'''
        super().__init__()

        #  输出变成 Advantage  和  Value， Advantage的维度是act_dim, V(s) 是1
        #  A(s,a)类似 Q(s,a) 在离散动作空间中， 这种动作价值直接输出 a 维度。
        self.useconve = useconve
        if self.useconve:
            # 如果使用卷及网络，那就用conv层
            self.conv_net, self.logits_net = conv_mlp(obs_dim, act_dim + 1)
        else:
            self.logits_net = mlp([obs_dim[0]] + list(hidden_sizes) + [act_dim + 1], activation)   # 把obs_dim[0] 防在这里是最合适的
    
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
```

## 3.4  her算法

适用于dict形式的gym环境。
