import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from net import opt_cri_arch
from replay_buffer import replay_buffer
import math


class option_critic(object):
    def __init__(self, env, episode, exploration, update_freq, freeze_interval, batch_size, capacity, learning_rate, option_num, gamma, termination_reg, epsilon_init, decay, epsilon_min, entropy_weight, conv, cuda, render, save_path=None):
        self.env = env
        self.episode = episode
        self.exploration = exploration
        self.update_freq = update_freq
        self.freeze_interval = freeze_interval
        self.batch_size = batch_size
        self.capacity = capacity
        self.learning_rate = learning_rate
        self.option_num = option_num
        self.gamma = gamma
        self.termination_reg = termination_reg
        self.epsilon_init = epsilon_init
        self.decay = decay
        self.epsilon_min = epsilon_min
        self.entropy_weight = entropy_weight
        self.conv = conv
        self.cuda = cuda
        self.render = render
        self.save_path = save_path

        if not self.conv:
            self.observation_dim = self.env.observation_space.shape[0]
        else:
            self.observation_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n
        self.epsilon = lambda x: self.epsilon_min + (self.epsilon_init - self.epsilon_min) * math.exp(- x / self.decay)
        self.net = opt_cri_arch(self.observation_dim, self.action_dim, self.option_num, self.conv)
        self.prime_net = opt_cri_arch(self.observation_dim, self.action_dim, self.option_num, self.conv)
        if self.cuda:
            self.net = self.net.cuda()
            self.prime_net = self.prime_net.cuda()
        self.prime_net.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.buffer = replay_buffer(self.capacity)
        self.count = 0
        self.weight_reward = None

    def compute_critic_loss(self, ):
        if self.cuda:
            torch.FloatTensor = torch.cuda.FloatTensor
            torch.LongTensor = torch.cuda.LongTensor
        observations, options, rewards, next_observations, dones = self.buffer.sample(self.batch_size)

        observations = torch.FloatTensor(observations)
        options = torch.LongTensor(options)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_observations = torch.FloatTensor(next_observations)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        states = self.net.get_state(observations)
        q_values = self.net.get_q_value(states)

        prime_next_states = self.prime_net.get_state(next_observations)
        prime_next_q_values = self.prime_net.get_q_value(prime_next_states)

        next_states = self.net.get_state(next_observations)
        next_q_values = self.net.get_q_value(next_states)

        next_betas = self.net.get_termination(next_states)
        next_beta = next_betas.gather(1, options.unsqueeze(1))

        target_q_omega = rewards + self.gamma * (1 - dones) * ((1 - next_beta) * prime_next_q_values.gather(1, options.unsqueeze(1)) + next_beta * prime_next_q_values.max(1)[0].unsqueeze(1))
        td_error = (target_q_omega.detach() - q_values.gather(1, options.unsqueeze(1))).pow(2).mean()
        return td_error

    def compute_actor_loss(self, obs, option, log_prob, entropy, reward, done, next_obs,):
        if self.cuda:
            torch.FloatTensor = torch.cuda.FloatTensor
            torch.LongTensor = torch.cuda.LongTensor
        obs = torch.FloatTensor(np.expand_dims(obs, 0))
        next_obs = torch.FloatTensor(np.expand_dims(next_obs, 0))

        state = self.net.get_state(obs)
        next_state = self.net.get_state(next_obs)
        prime_next_state = self.prime_net.get_state(next_obs)

        next_beta = self.net.get_termination(next_state)[:, option]
        beta = self.net.get_termination(state)[:, option]

        q_value = self.net.get_q_value(state)
        next_q_value = self.net.get_q_value(next_state)
        prime_next_q_value = self.prime_net.get_q_value(next_state)

        gt = reward + self.gamma * (1 - done) * ((1 - next_beta) * prime_next_q_value[:, option] + next_beta * prime_next_q_value.max(1)[0].unsqueeze(0))

        termination_loss = next_beta * ((next_q_value[:, option] - next_q_value.max(1)[0].unsqueeze(1)).detach() + self.termination_reg) * (1 - done)

        policy_loss = -log_prob * (gt - q_value[:, option]).detach() - self.entropy_weight * entropy
        actor_loss = termination_loss + policy_loss
        return actor_loss

    def run(self):
        if self.cuda:
            torch.FloatTensor = torch.cuda.FloatTensor
            torch.LongTensor = torch.cuda.LongTensor

        for i in range(self.episode):
            obs = self.env.reset()
            if self.render:
                self.env.render()
            total_reward = 0
            greedy_option = self.net.get_option(self.net.get_state(torch.FloatTensor(np.expand_dims(obs, 0))))
            termination = True
            current_option = 0
            while True:
                epsilon = self.epsilon(self.count)
                if termination:
                    current_option = random.choice(list(range(self.option_num))) if epsilon > random.random() else greedy_option
                action, log_prob, entropy = self.net.get_action(self.net.get_state(torch.FloatTensor(np.expand_dims(obs, 0))), current_option)
                next_obs, reward, done, info = self.env.step(action)
                self.count += 1
                total_reward += reward
                self.buffer.store(obs, current_option, reward, next_obs, done)
                if self.render:
                    self.env.render()

                termination, greedy_option = self.net.get_option_termination(self.net.get_state(torch.FloatTensor(np.expand_dims(next_obs, 0))), current_option)

                if len(self.buffer) > self.exploration:
                    loss = 0
                    loss += self.compute_actor_loss(obs, current_option, log_prob, entropy, reward, done, next_obs)

                    if self.count % self.update_freq == 0:
                        loss += self.compute_critic_loss()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if self.count % self.freeze_interval == 0:
                        self.prime_net.load_state_dict(self.net.state_dict())

                obs = next_obs

                if done:
                    if not self.weight_reward:
                        self.weight_reward = total_reward
                    else:
                        self.weight_reward = 0.99 * self.weight_reward + 0.01 * total_reward

                    print('episode: {}  reward: {}  weight_reward: {:.2f}'.format(i + 1, total_reward, self.weight_reward))
                    break
        if self.save_path:
            torch.save(self.net, self.save_path)
