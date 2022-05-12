import numpy as np
import random
from collections import deque

class replay_buffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def store(self, observation, option, reward, next_observation, done):
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)
        self.memory.append([observation, option, reward, next_observation, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observation, option, reward, next_observation, done = zip(* batch)
        return np.concatenate(observation, 0), option, reward, np.concatenate(next_observation, 0), done

    def __len__(self):
        return len(self.memory)