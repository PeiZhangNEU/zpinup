import gym
from replay_buffer import replay_buffer
from net import opt_cri_arch
from model import option_critic
import torch
import os


if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    # env = env.unwrapped                   # 对于不会返回done的程序，不要使用unwrapped！
    # cuda = torch.cuda.is_available()
    cuda = False
    os.makedirs('alogos/HRL/option_critic/option_critic_continues/model', exist_ok=True)
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
        save_path='alogos/HRL/option_critic/option_critic_continues/model/pendulum.pkl'
    )
    test.run()