import numpy as np
import matplotlib.pyplot as plt
import gym

ENV = 'CartPole-v1'
NUM_DIGITIZED = 6
GAMMA = 0.99 #decrease rate
ETA = 0.5 #learning rate
MAX_STEPS = 200 #steps for 1 episode
NUM_EPISODES = 2000 #number of episodes

class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)
    
    #update the Q function
    def update_Q_function(self, observation, action, reward, observation_next):
        self.brain.update_Q_table(
            observation, action, reward, observation_next)
     
    #get the action
    def get_action(self, observation, step):
        action = self.brain.decide_action(observation, step)
        return action
    
    def get_test_action(self, observation):
        action = self.brain.test_decide_action(observation)
        return action

class Brain:
    #do Q-learning
    
    def  __init__(self, num_states, num_actions):
        self.num_actions = num_actions #the number of CartPole actions
    
        #create the Q table, row is the discrete state(digitized state^number of variables), column is action(left, right)
        self.q_table = np.random.uniform(low=0, high=1, size=(NUM_DIGITIZED**num_states, num_actions)) #uniform distributed sample with size
    
    def bins(self, clip_min, clip_max, num):
        #convert continous value to discrete value
        return np.linspace(clip_min, clip_max, num + 1)[1: -1]   #num of bins needs num+1 value
    
    def digitize_state(self, observation):
        #get the discrete state in total 1296 states
        cart_pos, cart_v, pole_angle, pole_v = observation
        
        digitized = [
            np.digitize(cart_pos, bins = self.bins(-2.4, 2.4, NUM_DIGITIZED)),
            np.digitize(cart_v, bins=self.bins(-3.0, 3.0, NUM_DIGITIZED)),
            np.digitize(pole_angle, bins=self.bins(-0.5, 0.5, NUM_DIGITIZED)), #angle represent by radian
            np.digitize(pole_v, bins=self.bins(-2.0, 2.0, NUM_DIGITIZED))
        ]
        
        return sum([x* (NUM_DIGITIZED**i) for i, x in enumerate(digitized)])
    
    def update_Q_table(self, observation, action, reward, observation_next):
        state = self.digitize_state(observation)
        state_next = self.digitize_state(observation_next)
        Max_Q_next = max(self.q_table[state_next][:])
        self.q_table[state, action] = self.q_table[state, action] + \
            ETA * (reward + GAMMA * Max_Q_next - self.q_table[state, action])
        
    def decide_action(self, observation, episode):
        #epsilon-greedy
        state = self.digitize_state(observation)
        epsilon = 0.5 * (1 / (episode + 1))
        
        if epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q_table[state][:])
        else:
            action = np.random.choice(self.num_actions)
            
        return action
    
    def test_decide_action(self, observation):
        #no greedy
        state = self.digitize_state(observation)
        
        action = np.argmax(self.q_table[state][:])

            
        return action

class Environment:
    
    def __init__(self):
        self.env = gym.make(ENV)
        num_states = self.env.observation_space.shape[0] #4
        num_actions = self.env.action_space.n #2
        self.agent = Agent(num_states, num_actions) #create the agent
    
    def run(self):
        episodes_steps = []

        for episode in range(NUM_EPISODES):   #1000 episodes
            observation = self.env.reset()  #initialize environment

            for step in range(MAX_STEPS):   #steps in one episode
                action = self.agent.get_action(observation, episode) 
                observation_next, _, done, _ = self.env.step(action) #reward and info not need
                
                # reward shaping 
                # 这个reward shaping 方法，使得一回合结束之后r为 0 -1 或者1
                #get reward
                if done: #step > 200 or larger than angle
                    if step < 195:
                        reward = -1  #give punishment if game over less than last step
                        complete_episodes = 0  #game over less than 195 step then reset
                    else:   
                        reward = 1  
                        complete_episodes += 1  
                else:
                    reward = 0   #until done, reward is 0 
                #update Q table
                self.agent.update_Q_function(observation, action, reward, observation_next)
                
                #update observation
                observation = observation_next

                if done:
                    break
            
            print('this episode is {} and ep_rd is{}'.format(episode, step+1))
            episodes_steps.append(step+1)

        self.save_policy()
        plt.plot(episodes_steps)
        plt.show()
    
    def test_agent(self):
        
        for episode in range(10):   #1000 episodes
            observation = self.env.reset()  #initialize environment
            episode_r = 0
            for step in range(MAX_STEPS):   #steps in one episode
                action = self.agent.get_test_action(observation) 
                # action = self.agent.get_action(observation, episode) 
                observation_next, reward, done, _ = self.env.step(action)   
                self.env.render()
                episode_r += reward
                observation = observation_next 
                if done:
                    break
            print('this episode is {} and ep_rd is{}'.format(episode, episode_r))



    def save_policy(self):
        np.save("q_table.npy", self.agent.brain.q_table)

    def load_policy(self):
        self.agent.brain.q_table = np.load("alogos/cartpole_qlearning/q_table.npy")


if __name__ == '__main__':
    env = Environment()
    env.load_policy()
    env.test_agent()