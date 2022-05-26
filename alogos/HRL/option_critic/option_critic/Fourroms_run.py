import gym
from replay_buffer import replay_buffer
from net import opt_cri_arch
from model import option_critic
from four_rooms import FourRooms
import torch
import os


if __name__ == '__main__':
    env = FourRooms()
    cuda = torch.cuda.is_available()
    os.makedirs('alogos/HRL/option_critic/option_critic/model', exist_ok=True)
    test = option_critic(
        env=env,
        episode=2000,
        exploration=1000,
        update_freq=4,
        freeze_interval=200,
        batch_size=64,
        capacity=100000,
        learning_rate=3e-4,
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
    print(test.net)
    test.run()