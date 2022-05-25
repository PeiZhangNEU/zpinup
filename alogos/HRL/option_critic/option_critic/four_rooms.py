import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from gym.spaces import Discrete

class FourRooms:

	def __init__(self):
		layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
		self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])
		self.iner_steps = 0
		
		# Four possible actions
		# 0: UP
		# 1: DOWN
		# 2: LEFT
		# 3: RIGHT
		self.action_space = Discrete(4)
		self.observation_space = np.zeros(np.sum(self.occupancy == 0))
		self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]

		# Random number generator
		self.rng = np.random.RandomState(1234)

		self.tostate = {}
		statenum = 0
		for i in range(13):
			for j in range(13):
				if self.occupancy[i,j] == 0:
					self.tostate[(i,j)] = statenum
					statenum += 1
		self.tocell = {v:k for k, v in self.tostate.items()}


		self.goal = 65 # East doorway
		self.init_states = list(range(self.observation_space.shape[0]))
		self.init_states.remove(self.goal)


	def render(self, show_goal=True):
		current_grid = np.array(self.occupancy)
		current_grid[self.current_cell[0], self.current_cell[1]] = -1
		if show_goal:
			goal_cell = self.tocell[self.goal]
			current_grid[goal_cell[0], goal_cell[1]] = -1
		return current_grid

	def reset(self):
		self.iner_steps = 0
		state = self.rng.choice(self.init_states)
		self.current_cell = self.tocell[state]
		state_obs = np.zeros(self.observation_space.shape[0])
		state_obs[state] = 1.0
		return state_obs

	def check_available_cells(self, cell):
		available_cells = []

		for action in range(self.action_space.n):
			next_cell = tuple(cell + self.directions[action])

			if not self.occupancy[next_cell]:
				available_cells.append(next_cell)

		return available_cells
		

	def step(self, action):
		'''
		Takes a step in the environment with 2/3 probability. And takes a step in the
		other directions with probability 1/3 with all of them being equally likely.
		'''	
		self.iner_steps += 1
		next_cell = tuple(self.current_cell + self.directions[action])

		if not self.occupancy[next_cell]:

			if self.rng.uniform() < 1/3:
				available_cells = self.check_available_cells(self.current_cell)
				self.current_cell = available_cells[self.rng.randint(len(available_cells))]

			else:
				self.current_cell = next_cell

		state = self.tostate[self.current_cell]
		state_obs = np.zeros(self.observation_space.shape[0])
		state_obs[state] = 1.0

		# When goal is reached, it is done
		done = (state == self.goal) or (self.iner_steps==20)  # 20步走不到也是失败


		return state_obs, float(state == self.goal), done, None

if __name__ == '__main__':
    env = FourRooms()
    s = env.reset()
    plt.imshow(env.render(show_goal=True), cmap='Blues')
    plt.axis('off')
    plt.show()
    print(env.observation_space.shape)
    for i in range(300):
        print(s)
        clear_output(True)
        plt.imshow(env.render(show_goal=True), cmap='Blues')
        plt.axis('off')
        plt.show()

        a = np.random.randint(0,4)
        s_, r, d, _ = env.step(a)
        s = s_
        if d:
            print(env.iner_steps)
            break
        