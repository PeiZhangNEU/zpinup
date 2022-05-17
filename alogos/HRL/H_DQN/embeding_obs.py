import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym

env = gym.make('Taxi-v3')

obs = env.reset()
obs = np.array([obs, obs])
obs = torch.LongTensor(obs)  # obs 必须是Long tensor！

layers = []
layers.append(nn.Embedding(500, 10))
layers.append(nn.Linear(10, 64))
layers.append(nn.Linear(64, 64))
layers.append(nn.Linear(64, 500))
model = nn.Sequential(*layers)

print(model(obs).shape)
