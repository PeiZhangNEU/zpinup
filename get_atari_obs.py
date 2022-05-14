import gym
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch


env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')  # atari游戏在这里设置完rendermode之后在下面不需要再 env.render()
## atari game env原本的obs形状为不固定的比如 [210, 160, 3] ，但pytorch的图表格式channel在最前 
## 所以使用deepmind开放的wrap可以把atari env 解构，然后把 obs 形状转换为一样的形状 [1, 84, 84]， 在对atari环境进行卷及网络的时候一定要首先对环境wrapper
env = wrap_deepmind(env)
env = wrap_pytorch(env)

obs = env.reset()
t = 1
while True:
    t += 1
    a = env.action_space.sample()
    obs, r, d,_ = env.step(a)
    print(obs==0)
    if d:
        break
print( t)