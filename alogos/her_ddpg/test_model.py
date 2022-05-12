import os
import os, sys
# sys.path.append('/home/zp/ompx_project/open_manipulator_gym')
import torch
from models import actor
from arguments import Args
import numpy as np
import gym
import time
# from bmirobot_env.bmirobot_push_F import bmirobotGympushEnv as bmenv
#加载训练好的模型 数据
model_path = "data/her_ddpg/FetchPush-v1/123_False1_model.pt"
actions = []
observations = []
a_goals, d_goals = [], []
infos = []

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

if __name__ == '__main__':
    args = Args()
    # load the model param
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    o_mean, o_std, g_mean, g_std, model,= torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment+
    env = gym.make(args.train_type)
    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0],
                  'goal': observation['desired_goal'].shape[0],
                  'action': env.action_space.shape[0],
                  'action_max': env.action_space.high[0],
                  }
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    is_successes = 0
    for i in range(args.demo_length):
        observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        ag = observation['achieved_goal']
        g = observation['desired_goal']
        episodeAcs, episodeObs, episodeInfo = [], [], []
        episodeAg, episodeDg = [], []

        episodeObs.append(obs.copy())
        episodeAg.append(ag.copy())
        max_episode_steps=50
        success_time = 0
        for t in range(max_episode_steps):
            env.render()
            time.sleep(0.05)
            #env.render()
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            # env.render()
            obs = observation_new['observation']
            ag = observation_new['achieved_goal']
            g = observation_new['desired_goal']
            episodeAcs.append(action)
            episodeObs.append(obs)
            episodeAg.append(ag)
            episodeDg.append(g)
            episodeInfo.append(info)
            if info['is_success']==1:
                success_time+=1
                if success_time==10:
                    is_successes+=1
                    break
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
    print('demo test success rate:', is_successes/args.demo_length)

