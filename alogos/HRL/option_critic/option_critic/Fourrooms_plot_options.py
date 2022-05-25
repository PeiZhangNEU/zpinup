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
    
    obs = env.reset()
    options_of_obs = []
    grid = np.array(env.occupancy)
    for j in range(104):  # j 就是 obs 的state num
        obs = np.zeros(104)
        obs[j] = 1.0   # 穷举出每一种obs
        greedy_option = test.net.get_option(test.net.get_state(torch.FloatTensor(np.expand_dims(obs, 0))))  # 根据现在的状态得到q表，用argmax得到最好的option
        greedy_option += 5
        options_of_obs.append(greedy_option)
        # 根据目前的state num 得到目前的current cell
        
        current_cell = env.tocell[j]
        grid[current_cell[0], current_cell[1]] = greedy_option
    print(options_of_obs)
    plt.imshow(grid, cmap='Blues')
    plt.colorbar()
    plt.axis('off')
    plt.show()
