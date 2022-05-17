# 必须要把根目录的路径添加进来！
import sys
sys.path.append('/home/zp/zpinup/')   
import os
import numpy as np
import torch
import gym
import time
import csv
from alogos.HRL.H_DQN.hdqn import h_dqn

def int_to_array(input_obs):
    '''
    把0-499的整数转换为 [1,] 的array
    '''
    obs = np.array([input_obs])
    return obs


def train(env_fn, ac_kwargs=dict(),  seed=0, 
            epochs=5000, batch_size=64,
            update_every=1, update_times=1,
            buffer_size_meta=int(4e4), buffer_size=int(4e4),
            gamma=0.99, epsilon=0.8,
            pi_lr=3e-4, delay_up=0.995, 
            num_test_episodes=10, max_ep_len=200,save_freq=1,
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

        agent = h_dqn(env_fn, ac_kwargs=ac_kwargs, device=device, 
                      buffer_size_meta=buffer_size_meta, buffer_size=buffer_size,
                      gamma=gamma, epsilon=epsilon, pi_lr=pi_lr, delay_up=delay_up,
                      num_test_episodes=num_test_episodes, max_ep_len=max_ep_len)
        env = env_fn()

        # 过程数据
        rolling_intrinsic_rewards = []
        rolling_env_rewards = []
        goals_seen = []
        controller_learnt_enough = False
        controller_actions = []

        # 随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 准备开始主循环
        t = 0
        start_time = time.time()

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
        
        for i in range(epochs):   # 最外循环

            agent.meta_controller.policy.epsilon = agent.meta_controller.policy.epsilon - agent.meta_controller.policy.epsilon * 0.001   # 每一个epoch， 减小一下epsilon， 这样Epsilon会不会减小的太快了？
            if agent.meta_controller.policy.epsilon <= 0.02:
                agent.meta_controller.policy.epsilon = 0.02
            
            agent.controller.policy.epsilon = agent.controller.policy.epsilon - agent.controller.policy.epsilon * 0.001   # 每一个epoch， 减小一下epsilon， 这样Epsilon会不会减小的太快了？
            if agent.controller.policy.epsilon <= 0.02:
                agent.controller.policy.epsilon = 0.02

            # 初始化
            env_state = int_to_array(env.reset())   # s
            next_state = None
            action = None
            reward = None
            done = False
            cumulative_meta_controller_reward = 0
            episode_over = False
            subgoal_achieved = False
            total_episode_score_so_far = 0
            meta_controller_steps = 0
            
            # 分层循环
            episode_steps = 0

            while not episode_over:   # while not 不需要 break 标志，会自动跳出
                episode_intrinsic_rewards = []
                meta_controller_state = env_state  # 现在env的obs  [1,]  s0 \gets s
                subgoal = int_to_array(agent.meta_controller.get_action(meta_controller_state))    # 更新g
                # print(subgoal)
                goals_seen.append(subgoal)
                subgoal_achieved = False
                state = np.concatenate((env_state, subgoal))  # {s, g}
                cumulative_meta_controller_reward = 0         # F \gets 0

                while not (episode_over or subgoal_achieved):  # while not 不需要 break 标志，会自动跳出
                    # 执行动作得到环境的反馈
                    action =  agent.controller.get_action(state)
                    controller_actions.append(action)
                    env_next_state, env_reward, env_done, _ = env.step(action)   # r = env_reward
                    env_next_state = int_to_array(env_next_state)
                    total_episode_score_so_far += env_reward
                    # 更新数据
                    episode_over = env_done
                    # 更新controller 的 data
                    next_state = np.concatenate((env_next_state, subgoal))
                    subgoal_achieved = (env_next_state==subgoal) # 判断现在是否达到目标了
                    f = 1.0 * subgoal_achieved                   # f = 1 * success
                    done = subgoal_achieved or episode_over
                    # 更新metacontroller 的 data
                    cumulative_meta_controller_reward += f       # F \gets F + f 
                    if done:
                        meta_controller_next_state = env_next_state   # 如果回合结束，更新 meta_control_state，一个回合中goal是一样的  

                    # 向controller的buffer中存数据
                    agent.controller.buffer.store(state, action, env_reward, next_state, done)
                    # 训练controller
                    if agent.controller.buffer.size>batch_size and t % update_every == 0:
                        for _ in range(update_times):
                            batch = agent.controller.buffer.sample_batch(batch_size)
                            agent.controller.update(data=batch)
                    env_state = env_next_state                   # 更新环境的state
                    state = next_state                           # 更新 state， 不断更新的一个回合内也更新
                    t += 1 # 总步数+1
                    episode_intrinsic_rewards.append(f)          

                # 向 meta_controller 的buffer中存数据
                agent.meta_controller.buffer.store(meta_controller_state, subgoal, cumulative_meta_controller_reward, meta_controller_next_state, episode_over)
                meta_controller_steps += 1
                if agent.meta_controller.buffer.size>batch_size and t % update_every == 0:
                        for _ in range(update_times):
                            batch2 = agent.meta_controller.buffer.sample_batch(batch_size)
                            agent.meta_controller.update(data=batch2)
            
            if i % 100 == 0:
                print(i)
                torch.save(agent.meta_controller.policy, model_dir + '/hdqn_' + str(i) + '_meta_policy.pt')
                torch.save(agent.controller.policy, model_dir + '/hdqn_' + str(i) + '_policy.pt')
                print(" ")
                print("Intrinsic Rewards -- {} -- ".format(np.mean(rolling_intrinsic_rewards[-10:])))
                print("Average controller action -- {} ".format(np.mean(controller_actions[-10:])))
                print("Latest subgoal -- {}".format(goals_seen[-1]))
                print('total_reward_this_epoch {}'.format(total_episode_score_so_far))
                print('epsilon {}'.format(agent.meta_controller.policy.epsilon))
                print('avg rewards {}'.format(np.mean(rolling_env_rewards[-10:])))
            rolling_intrinsic_rewards.append(np.sum(episode_intrinsic_rewards))
            rolling_env_rewards.append(total_episode_score_so_far)

if __name__ == '__main__':
    # 设置超参数
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Taxi-v3')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50000)  # 一共训练了50000轮   
    parser.add_argument('--exp_name', type=str, default='hdqn_taxi')
    args = parser.parse_args()

    # 执行训练过程
    train(lambda : gym.make(args.env),
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, epochs=args.epochs, expname=args.exp_name, use_gpu=False)
