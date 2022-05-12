# 必须要把根目录的路径添加进来！
import sys
sys.path.append('/home/zp/zpinup/')

import torch
import gym
import time

def load_model(path):
    '''把模型加载成cpu形式'''
    model = torch.load(path, map_location=torch.device('cpu'))
    return model

def get_action(model, x):
    '''因为model的act，需要传入tensor 的obs，这里写个函数转化'''
    with torch.no_grad():
        x = torch.as_tensor(x, dtype=torch.float32)
        action = model.act(x)
    return action

def test(path, env_name, render=True, num_episodes=2000, max_ep_len=1000):
    '''载入模型并且测试'''
    policy = load_model(path)  # 这个载入的policy和logger的save的东西有关
                               # 我save的是ActorCritic这个类，包括类的方法也保留
    env = gym.make(env_name)



    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(policy, o)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1


if __name__ == '__main__':
    test('data/vpg_hopper/vpg_50_ac.pt','Hopper-v2')
    # test('data/vpg_cartpole/vpg_cartpole_s0/pyt_save/model.pt','CartPole-v0')