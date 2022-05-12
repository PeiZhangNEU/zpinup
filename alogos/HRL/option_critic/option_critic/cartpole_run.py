import gym
from replay_buffer import replay_buffer
from net import opt_cri_arch
from model import option_critic
import torch
import os


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    # cuda = torch.cuda.is_available()
    cuda = False                               # 小型网络用cpu更快！
    os.makedirs('./model', exist_ok=True)
    test = option_critic(
        env=env,
        episode=1000,
        exploration=1000,
        update_freq=4,
        freeze_interval=200,
        batch_size=128,
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
        render=True,
        save_path='./model/cartpole.pkl'
    )
    test.run()