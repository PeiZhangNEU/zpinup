import gym
from replay_buffer import replay_buffer
from net import opt_cri_arch
from model import option_critic
import torch
import os
import numpy as np
import time
from IPython.display import clear_output
from four_rooms import FourRooms
import matplotlib.pyplot as plt


if __name__ == '__main__':
    env = FourRooms()
    cuda = torch.cuda.is_available()
    os.makedirs('alogos/HRL/option_critic/option_critic/model', exist_ok=True)
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
        save_path='alogos/HRL/option_critic/option_critic/model/fourrooms.pkl'
    )
    test.net = torch.load(test.save_path, map_location='cpu')
    print(test.net)
    

    # # 使用贪心策略不断切换我们的策略mu
    # all_reward = 0
    # for j in range(20):
    #     obs = env.reset()
    #     total_reward = 0
    #     greedy_option = test.net.get_option(test.net.get_state(torch.FloatTensor(np.expand_dims(obs, 0))))  # 根据现在的状态得到q表，用贪心策略得到最好的option
    #     termination = False   # 测试的时候不使用option跳变器，因为一直使用的都是 greedy option！跳变的目的是探索和优化两个策略，最终我们要找到每个状态最适合的mu策略
    #     current_option = 0
    #     for i in range(200):
    #         current_option = greedy_option
    #         action, log_prob, entropy = test.net.get_action(test.net.get_state(torch.FloatTensor(np.expand_dims(obs, 0))), current_option)
    #         next_obs, reward, done, info = test.env.step(action)
    #         total_reward += reward
    #         env.render()
    #         time.sleep(1/60)
    #         # 根据下个状态决定是否要跳转option, 并且得到下一个状态对应的greedy option
    #         termination, greedy_option = test.net.get_option_termination(test.net.get_state(torch.FloatTensor(np.expand_dims(next_obs, 0))), current_option)
    #         termination = False
    #         if done:
    #             break
    #         obs = next_obs
    #     print(total_reward)
    #     all_reward += total_reward
    # print('all_ave_r', all_reward/20)


    ## 一直使用第一个option
    # for j in range(10):
    #     obs = env.reset()
    #     current_option = 0
    #     total_reward = 0
    #     for i in range(200):
    #         action, log_prob, entropy = test.net.get_action(test.net.get_state(torch.FloatTensor(np.expand_dims(obs, 0))), current_option)
    #         next_obs, reward, done, info = test.env.step(action)
    #         total_reward += reward
    #         env.render()
    #         time.sleep(1/60)
    #         if done:
    #             break
    #         obs = next_obs
    #     print(total_reward)


    ## 使用Beta网络，按概率判断是否切换argmax得到的option, 这个最好！epsilon贪心策略 和 贪心策略是不同的， epsilon贪心策略，是有较大的概率选择argmax，而小概率随机！贪心策略指argmax
    all_reward = 0
    for j in range(20):
        obs = env.reset()
        total_reward = 0
        greedy_option = test.net.get_option(test.net.get_state(torch.FloatTensor(np.expand_dims(obs, 0))))  # 根据现在的状态得到q表，用argmax得到最好的option
        termination = False   
        current_option = 0
        
        for i in range(50):
            if termination:
                current_option = greedy_option   # 不再使用epsilon，而是直接等于贪心option
            action, log_prob, entropy = test.net.get_action(test.net.get_state(torch.FloatTensor(np.expand_dims(obs, 0))), current_option)
            next_obs, reward, done, info = test.env.step(action)
            total_reward += reward

            # clear_output(True)
            print('now option is:',current_option)
            plt.imshow(env.render(show_goal=True), cmap='Blues')
            plt.axis('off')
            plt.show()
            plt.close()

            time.sleep(1/60)
            # 根据下个状态决定是否要跳转option, 并且得到下一个状态对应的greedy option
            termination, greedy_option = test.net.get_option_termination(test.net.get_state(torch.FloatTensor(np.expand_dims(next_obs, 0))), current_option)
            if done:
                break
            obs = next_obs
        print(total_reward)
        all_reward += total_reward
    print('all_ave_r', all_reward/20)
