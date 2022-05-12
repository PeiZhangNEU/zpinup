import gym
from replay_buffer import replay_buffer
from net import opt_cri_arch
from model import option_critic
import torch
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import os


if __name__ == '__main__':
    env_id = "PongNoFrameskip-v4"
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)
    cuda = torch.cuda.is_available()
    os.makedirs('./model', exist_ok=True)
    test = option_critic(
        env=env,
        episode=3000,
        exploration=1000,
        update_freq=4,
        freeze_interval=200,
        batch_size=32,
        capacity=100000,
        learning_rate=1e-4,
        option_num=4,
        gamma=0.99,
        termination_reg=0.01,
        epsilon_init=1.,
        decay=10000,
        epsilon_min=0.01,
        entropy_weight=1e-2,
        conv=True,
        cuda=cuda,
        render=False,
        save_path='./model/pong.pkl'
    )
    test.run()