# 必须要把根目录的路径添加进来！
import sys
sys.path.append('/home/zp/zpinup/')
import numpy as np
import torch
import gym
import time

def int_to_array(input_obs):
    '''
    把0-499的整数转换为 [1,] 的array
    '''
    obs = np.array([input_obs])
    return obs

def load_model(path):
    '''把模型加载成cpu形式'''
    model = torch.load(path, map_location=torch.device('cpu'))
    return model

# 这里载入的只是ac，get_action函数在ddpg类里面，需要重新写一下
def get_action(model, x):
    '''因为model的act，需要传入tensor 的obs，这里写个函数转化
    on-policy和off-policy的ac都有act函数。
    '''
    with torch.no_grad():
        x = torch.as_tensor(x, dtype=torch.long)
        action = model.act(x)                         # 这里是 core 里面的actorCritic的函数 act，是给的确定性动作
    return action

def test(path1, path2, env_name, render=True, num_episodes=1, max_ep_len=200):
    '''载入模型并且测试'''
    policy = load_model(path1)  # 这个载入的policy和logger的save的东西有关
                               # 我save的是ActorCritic这个类，包括类的方法也保留

    meta_policy = load_model(path2)

    print(policy.pi)
    print(meta_policy.pi)
    env = gym.make(env_name)

    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    o = int_to_array(o)
    while n < num_episodes:
        if render:
            env.render()
        subgoal = get_action(meta_policy, o)
        subgoal = int_to_array(subgoal)

        state = np.concatenate((o, subgoal))
        action = get_action(policy, state)

        o, r, d, _ = env.step(action)
        o = int_to_array(o)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            o = int_to_array(o)
            n += 1


if __name__ == '__main__':
    test('data/hdqn_taxi/hdqn_9600_policy.pt','data/hdqn_taxi/hdqn_9600_meta_policy.pt','Taxi-v3')