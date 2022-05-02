# 必须要把根目录的路径添加进来！
import sys
sys.path.append('/home/zp/zpinup/')
import numpy as np
import torch
import gym
import time
import alogos.trpo.core as core
from alogos.trpo.trpo import *
import os
import csv



def train(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
          steps_per_epoch=2048, epochs=50, gamma=0.99, lam=0.97, delta=0.01,
          backtrack_iter=10, backtrack_coeff=1.0, backtrack_alpha=0.5,
          vf_lr=1e-3, train_v_iters=80, max_ep_len=1000,
          save_freq=10, use_gpu=False, expname='', 
          mode='TRPO'):
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

        agent = trpo(env_fn=env_fn, actor_critic=actor_critic, ac_kwargs=ac_kwargs, 
                steps_per_epoch=steps_per_epoch,
                gamma=gamma, lam=lam, delta=delta, 
                vf_lr=vf_lr, train_v_iters=train_v_iters, 
                backtrack_iter=backtrack_iter, backtrack_coeff=backtrack_coeff, backtrack_alpha=backtrack_alpha,
                device=device,
                mode=mode)
        env = env_fn()

        # 随机种子
        seed = seed + 10000 
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 准备开始主循环
        start_time = time.time()
        o, ep_ret, ep_len = env.reset(), 0, 0

        # 主循环, 一个epoch运行了好多个回合，具体不计数，自动处理
        
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

        for epoch in range(epochs):
            for t in range(steps_per_epoch):  # buffer 的 max_size
                a, v, logp = agent.ac.step(torch.as_tensor(o, dtype=torch.float32).to(device))
                next_o, r, d, _ = env.step(a)
                ep_ret += r
                ep_len += 1

                # save到buffer并记录
                agent.buf.store(o, a, r, v, logp)       # 内循环一定会执行完，不会提前终止，所以buffer最后一定是满的
                agent.information['VVals'] = v

                # 更新obs
                o = next_o

                timeout = ep_len==max_ep_len              # 智能体可以不死执行到本回合结束
                terminal = d or timeout                   # 智能体死了done或者执行完了回合还没死，就terminal
                epoch_ended = t==steps_per_epoch-1  # 目前的步数t==epoch最大步数-1，也就是内循环运行结束了

                # 如果上述任意情况发生，eplen置0，但是t还是会继续累加的，t会一直达到maxsize才会跳出内循环
                if terminal or epoch_ended:
                    # 如果智能体一个epoch最大步数时，刚好没死
                    if epoch_ended and not(terminal):  
                        print('警告：轨迹在 %d step 被epoch截断.'%ep_len, flush=True)
                    # 如果智能体没死或者目前的步数达到了回合结束，那么就把最后
                    if timeout or epoch_ended:
                        _, v, _ = agent.ac.step(torch.as_tensor(o, dtype=torch.float32).to(device))
                    else:
                        v = 0
                    # 在一段轨迹结束时调用这个函数，填充到轨迹最后的last_value就是刚刚的v
                    agent.buf.finish_path(v)
                    if terminal:
                        # 仅仅在轨迹完成时保存Epret/Eplen
                        agent.information['Epret'] = ep_ret
                        agent.information['Eplen'] = ep_len
                    
                    o, ep_ret, ep_len = env.reset(), 0, 0
                
            # 每个epoch达到保存或者主循环结束
            if (epoch % save_freq==0) or (epoch == epochs-1):
                torch.save(agent.ac, model_dir + '/trpo_' + str(epoch) + '_ac.pt')
            
            # 进行TRPO的梯度更新
            agent.update()

            #==================================================================

            # 把一个epoch的内容写入csv
            # 把字典扩充了
            agent.information['Epoch'] = epoch
            agent.information['Total_Steps'] = (epoch+1)*steps_per_epoch
            header = list(agent.information.keys())
            datas = []
            datas.append(agent.information)
            with open(model_dir+str('/process.csv'), 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                if epoch == 0:
                    writer.writeheader()   # 只在第一次写入列名
                writer.writerows(datas)

            # 打印 字典所有的东西
            print('Epoch', epoch)
            print('Toltal_Steps', (epoch+1)*steps_per_epoch)
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
    # parser.add_argument('--env', type=str, default='CartPole-v0')
    # parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--env', type=str, default='Pendulum-v1')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=3000)     # 发现ppo的epoch应该少一点，回合内部应该多一点，不知道为什么到了后期奖励会骤减
    parser.add_argument('--epochs', type=int, default=1000)  # 一共训练了3e6次
    # parser.add_argument('--exp_name', type=str, default='trpo_cartpole')
    # parser.add_argument('--exp_name', type=str, default='trpo_hopper')
    parser.add_argument('--exp_name', type=str, default='trpo_pendulum')
    parser.add_argument('--mode', type=str, default='TRPO')   # 选择更新模式 'TRPO'还是'NPG'
    args = parser.parse_args()


    # 执行训练过程
    train(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        max_ep_len=200, # max_ep_len 需要和gym环境的本身的最大步长相同！Hopper是1000，Pendulum是200，这个值可以通过env._max_episode_steps获得
        use_gpu=True, expname=args.exp_name, mode=args.mode)