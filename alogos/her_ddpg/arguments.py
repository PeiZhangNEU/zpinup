import argparse
#argparse的库可以在命令行中传入参数并让程序运行
"""
Here are the param for the training

"""
class Args:
    def __init__(self):
        self.n_epochs = 400  # 50
        self.n_cycles = 50
        self.n_batches = 40
        self.save_interval = 5
        self.seed = 123  # 123
        self.num_workers = 19  # 1
        self.replay_strategy = 'future'
        self.clip_return = 50
        self.save_dir = 'data/her_ddpg/'      # 每次训练之前可以改一下名字，把保存的模型单独放置
        self.noise_eps = 0.01
        self.random_eps = 0.3
        self.buffer_size = 1e6*1/2
        self.replay_k = 4  # replay with k random states which come from the same episode as the transition being replayed and were observed after it
        self.clip_obs = 200
        self.batch_size = 256
        self.gamma = 0.98
        self.action_l2 = 1
        self.lr_actor = 0.001
        self.lr_critic = 0.001
        self.polyak = 0.95  # 软更新率
        self.n_test_rollouts = 25 #在训练时测试次数
        self.clip_range = 5
        self.demo_length = 25  # 20
        self.cuda = True
        self.num_rollouts_per_mpi = 3
        self.add_demo = False  # add demo data or not
        self.demo_name="bmirobot_1000_push_demo.npz"
        #self.demo_name="bmirobot_1000_pick_demo.npz"
        self.train_type = "FetchPush-v1" #or "pick" or 'reach' or 'slide' or 'obsnoise_push' or 'obsnoise_pick' or 'obsnoise_reach'
        self.Use_GUI =True  #GUI is for visualizing the training process
        self.env_name = 'FetchPush-v1'