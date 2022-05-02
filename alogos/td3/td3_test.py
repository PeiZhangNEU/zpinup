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

# 这里载入的只是ac，get_action函数在ddpg类里面，需要重新写一下
def get_action(model, x):
    '''因为model的act，需要传入tensor 的obs，这里写个函数转化
    on-policy和off-policy的ac都有act函数。
    '''
    with torch.no_grad():
        x = torch.as_tensor(x, dtype=torch.float32)
        action = model.act(x)                         # 这里是 core 里面的actorCritic的函数 act，是给的确定性动作
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
        # 载入的 ac 本身就有 get_action
        a = get_action(policy, o)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1


if __name__ == '__main__':
    test('data/td3_hopper/td3_4_ac.pt','Hopper-v2')