# 必须要把根目录的路径添加进来！
import sys
sys.path.append('/home/zp/zpinup/')
import numpy as np
import torch
import gym
import time
import alogos.sac_discrete.core as core
from alogos.sac_discrete.sac_discrete import *
import os
import csv

def train(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
         steps_per_epoch=4000, epochs=100, replay_size=int(5e3), gamma=0.99, 
         delayup=0.995, pi_lr=1e-3, q_lr=1e-3, alpha_lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
         update_after=1000, update_every=50, update_times=50, num_test_episodes=10, 
         max_ep_len=200, save_freq=1, expname='', use_gpu=False):
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

        agent = sac_discrete(env_fn, actor_critic=actor_critic, ac_kwargs=ac_kwargs, 
                    replay_size=replay_size, gamma=gamma, 
                    delayup=delayup, pi_lr=pi_lr, q_lr=q_lr, alpha_lr=alpha_lr, alpha=alpha,
                    num_test_episodes=num_test_episodes, max_ep_len=max_ep_len,
                    device=device)
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
            if t > start_steps:
                a = agent.get_action(o)    # 从分布采取动作，不使用直接的方式！
            else:
                a = env.action_space.sample()
            
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
            
            # 如果达到了更新步数，之后每隔50步就update 1次. 之前总是在多次更新之后出现None值，为什么，因为Update太多次，最后每次奖励都是200，导致梯度消失了！
            if t >= update_after and t % update_every == 0:
                batch = agent.buffer.sample_batch(batch_size)
                agent.update(data=batch)
            
            # 打印以及存储模型，以及测试模型
            if (t+1) % steps_per_epoch == 0:
                epoch = (t+1) // steps_per_epoch   # 除完向下取整

                # 存储模型
                if (epoch % save_freq == 0) or (epoch == epochs):
                    torch.save(agent.ac, model_dir + '/sac_dis_' + str(epoch) + '_ac.pt')
                
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
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=250)  # 一共训练了3e6次
    parser.add_argument('--exp_name', type=str, default='sac_dis_cartpole')
    args = parser.parse_args()


    # 执行训练过程
    train(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, epochs=args.epochs, expname=args.exp_name, use_gpu=True)