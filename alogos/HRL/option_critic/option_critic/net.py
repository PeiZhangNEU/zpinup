import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_buffer import replay_buffer
import numpy as np


class opt_cri_arch(nn.Module):
    def __init__(self, observation_dim, action_dim, option_num, conv=False):
        super(opt_cri_arch, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.option_num = option_num
        self.conv = conv

        if not self.conv:
            self.feature = nn.Sequential(
                nn.Linear(self.observation_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )
        else:
            self.feature = nn.Sequential(
                nn.Conv2d(self.observation_dim[0], 32, 8, 4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1),
                nn.ReLU()
            )
            self.linear_feature = nn.Sequential(
                nn.Linear(self.feature_size(), 128),
                nn.ReLU()
            )

        self.q_value_layer = nn.Linear(128, self.option_num)

        self.termination_layer = nn.Linear(128, self.option_num)
        self.option_layer = nn.ModuleList([nn.Linear(128, self.action_dim) for _ in range(self.option_num)])

    def feature_size(self):
        tmp = torch.zeros(1, * self.observation_dim)
        return self.feature(tmp).view(1, -1).size(1)

    def get_state(self, observation):
        if not self.conv:
            return self.feature(observation)
        else:
            conv_feature = self.feature(observation).view(observation.size(0), -1)
            return self.linear_feature(conv_feature)

    def get_q_value(self, state):
        return self.q_value_layer(state)

    def get_option_termination(self, state, current_option):
        termination = self.termination_layer(state)[:, current_option].sigmoid()
        if self.training:   # 这个是nnMoudle的属性，一直为True
              option_termination = torch.distributions.Bernoulli(termination)    # 根据概率得到一个波努力分布
              option_termination = option_termination.sample()                   # 利用伯努里分布按照概率采样出一个0或者1！
        else:
            option_termination = (termination > 0.5)                             # 当概率大于0.5才发生转移
        q_value = self.get_q_value(state)
        next_option = q_value.max(1)[1].detach().item()
        return bool(option_termination), next_option                              # 把option_termination转换成bool再更新termination

    def get_termination(self, state):
        return self.termination_layer(state).sigmoid()

    def get_action(self, state, current_option):
        x = self.option_layer[current_option](state)
        prob = F.softmax(x, dim=1)
        dist = torch.distributions.Categorical(prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        action = action.detach().item()
        return action, log_prob, entropy

    def get_option(self, state):
        q_value = self.get_q_value(state)
        next_option = q_value.max(1)[1].detach().item()
        return next_option