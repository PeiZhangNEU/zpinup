# 必须要把根目录的路径添加进来！
import sys
sys.path.append('/home/zp/zpinup/')   
import os
import numpy as np
import torch
import gym
import time
import alogos.duelingddqn.core as core
import csv
from alogos.duelingddqn.duelingddqn import *

from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch

def train(env_fn, policy=core.MLPpolicy, ac_kwargs=dict(), seed=0, 
          steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, epsilon=0.5,
          pi_lr=1e-3,  batch_size=64, delay_up=0.995, useconve=False,
          update_after=1000, update_every=50, update_times=50,
          num_test_episodes=10, max_ep_len=1000, save_freq=1,
          expname='', use_gpu=False):
        '''
        训练函数，传入agent和env进行训练。主循环。
        '''
        # 训练时可以选择使用GPU或者是CPU
        if use_gpu:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                print("这台设备不支持gpu，自动调用cpu")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')

        agent = dqn(env_fn, policy=policy, ac_kwargs=ac_kwargs, 
                    replay_size=replay_size, gamma=gamma, epsilon=epsilon,
                    pi_lr=pi_lr, delay_up=delay_up, useconve=useconve,
                    num_test_episodes=num_test_episodes, max_ep_len=max_ep_len, device=device)
        env = env_fn()

        # 随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 准备开始主循环
        total_steps = steps_per_epoch * epochs   # 和on-policy不同，off-policy的算法用总步数进行训练，不分epoch了
        start_time = time.time()
        o, ep_ret, ep_len = env.reset(), 0, 0

        # 主循环,就一个循环，不需要on policy的那种麻烦的终止条件。

        # 创建目录======================================================
        model_dir = 'data/' + str(expname)
        if os.path.exists(model_dir):
            pass
        else:
            os.makedirs(model_dir)
        # 把之前的csv删除
        if os.path.exists(model_dir+str('/process.csv')):
            os.remove(model_dir+str('/process.csv'))
        # ==============================================================
        
        for t in range(total_steps): 
            # 在开始步数到达之前，只使用随机动作；达到开始步数之后，使用pi得到的加噪声的动作
            a = agent.get_action(o)
            
            # 执行环境交互
            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # 如果因为运行到末尾出现done，把done置为False
            d = False if ep_len == max_ep_len else d

            # 存数据到buffer
            agent.buffer.store(o, a, r, o2, d)

            # 非常重要，更新状态
            o = o2

            # 如果 d 或者 回合结束，则重置状态和计数器
            if d or (ep_len == max_ep_len):
                agent.information['Epret'] = ep_ret
                agent.information['Eplen'] = ep_len
                o, ep_ret, ep_len = env.reset(), 0, 0
            
            # 如果达到了更新步数，之后每隔50步就update50次
            if t >= update_after and t % update_every == 0:
                for _ in range(update_times):
                    batch = agent.buffer.sample_batch(batch_size)
                    agent.update(data=batch)
            
            # 打印以及存储模型，以及测试模型
            if (t+1) % steps_per_epoch == 0:
                epoch = (t+1) // steps_per_epoch   # 除完向下取整
                if epoch >= 30:  # 前30个回合自由探索一点？再进行下降？
                    agent.policy.epsilon = epsilon * (1 / (epoch-30 + 1))   # 每一个epoch， 减小一下epsilon， 这样Epsilon会不会减小的太快了？
                agent.information['Epsilon'] = agent.policy.epsilon

                # 存储模型
                if (epoch % save_freq == 0) or (epoch == epochs):
                    torch.save(agent.policy, model_dir + '/dqn_' + str(epoch) + '_policy.pt')
                
                # 测试目前的表现
                agent.test_agent()

                #==================================================================
                # 把一个epoch的内容写入csv
                # 把字典扩充了
                agent.information['Epoch'] = epoch
                agent.information['Total_Steps'] = t
                header = list(agent.information.keys())
                datas = []
                datas.append(agent.information)
                with open(model_dir+str('/process.csv'), 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=header)
                    if epoch == 1:
                        writer.writeheader()   # 只在第一次写入列名
                    writer.writerows(datas)
                # 打印 字典所有的东西
                print('Epoch', epoch)
                print('Toltal_Steps', t)
                print('==============================================')
                for i in agent.information.keys():
                    print(i+':')
                    print(agent.information[i])
                    print('------------------------------------')
                print('Time', time.time()-start_time)
                print('==============================================')
                print('\n')
                #====================================================================

if __name__ == '__main__':
    # 设置超参数
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Breakout-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=250)  # 一共训练了3e6次
    parser.add_argument('--exp_name', type=str, default='duelingdqn_breakout')
    args = parser.parse_args()

    # 执行训练过程
    # env = gym.make(args.env, render_mode='human')  # atari游戏在这里设置完rendermode之后在下面不需要再 env.render()
    env = gym.make(args.env)
    ## atari game env原本的obs形状为不固定的比如 [210, 160, 3] ，但pytorch的图表格式channel在最前 
    ## 所以使用deepmind开放的wrap可以把atari env 解构，然后把 obs 形状转换为一样的形状 [1, 84, 84]， 在对atari环境进行卷及网络的时候一定要首先对环境wrapper
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)
    train(lambda : env, policy=core.MLPpolicy,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, useconve=True,
        seed=args.seed, epochs=args.epochs, expname=args.exp_name, use_gpu=True)