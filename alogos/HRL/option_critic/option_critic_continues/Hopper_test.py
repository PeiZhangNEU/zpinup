import gym
from replay_buffer import replay_buffer
from net import opt_cri_arch
from model import option_critic
import torch
import os
import numpy as np
import time


if __name__ == '__main__':
    env = gym.make('Hopper-v3')
    # env = env.unwrapped                   # 对于不会返回done的程序，不要使用unwrapped！
    # cuda = torch.cuda.is_available()
    cuda = False
    os.makedirs('alogos/HRL/option_critic/option_critic_continues/model', exist_ok=True)
    test = option_critic(
        env=env,
        episode=1000,
        exploration=1000,
        update_freq=4,
        freeze_interval=200,
        batch_size=32,
        capacity=100000,
        learning_rate=1e-4,
        option_num=2,
        gamma=0.99,
        termination_reg=0.01,
        epsilon_init=1.,
        decay=10000,
        epsilon_min=0.01,
        entropy_weight=1e-2,
        conv=False,
        cuda=cuda,
        render=False,
        save_path='alogos/HRL/option_critic/option_critic_continues/model/hopper.pkl'
    )
    test.net = torch.load(test.save_path, map_location='cpu')
    print(test.net)
    
    ## 使用Beta网络，按概率切换贪心策略option, 这个最好！
    for j in range(10):
        obs = env.reset()
        total_reward = 0
        greedy_option = test.net.get_option(test.net.get_state(torch.FloatTensor(np.expand_dims(obs, 0))))  # 根据现在的状态得到q表，用贪心策略得到最好的option
        termination = False   
        current_option = 0
        
        for i in range(1000):
            if termination:
                current_option = greedy_option
            action, log_prob, entropy = test.net.get_action(test.net.get_state(torch.FloatTensor(np.expand_dims(obs, 0))), current_option)
            next_obs, reward, done, info = test.env.step(action)
            total_reward += reward
            env.render()
            time.sleep(1/60)
            # 根据下个状态决定是否要跳转option, 并且得到下一个状态对应的greedy option
            termination, greedy_option = test.net.get_option_termination(test.net.get_state(torch.FloatTensor(np.expand_dims(next_obs, 0))), current_option)
            if done:
                break
            obs = next_obs
        print(total_reward)
