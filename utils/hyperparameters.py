import torch as t
import math


class Config:
    def __init__(self):
        # 设置使用GPU
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        # 根目录
        self.root_dir = "/home/cbd109/Users/smy/DataSets/DRLImp/"
        # 模型存放目录
        self.model_dir = self.root_dir + "model/"
        # 数据目录
        self.data_dir = self.root_dir + "data/"
        # DQN
        # epsilon-greedy参数
        # 起始epsilon
        self.epsilon_start = 1.0
        # 终止epsilon
        self.epsilon_final = 0.01
        # epsilon衰退分母
        self.epsilon_decay = 30000
        # 指定帧的epsilon
        self.epsilon_by_frame = lambda frame_idx: self.epsilon_final + (
                self.epsilon_start - self.epsilon_final) * math.exp(-1. * frame_idx / self.epsilon_decay)
        # 折扣因子
        self.GAMMA = 0.99
        # 学习率
        self.LR = 1e-4

        # 内存相关
        # 目标网络更新频率
        self.TARGET_NET_UPDATE_FREQ = 1000
        # 经验回放存储区大小
        self.EXP_REPLAY_SIZE = 100000
        # 批次大小
        self.BATCH_SIZE = 32
        # self.BATCH_SIZE = 128

        # 达到learn_start才开始学习，未达到之前仅累计经验放入经验回放机制中
        self.LEARN_START = 10000
        self.MAX_FRAMES = 1000000

        # DQN_NSTEP所用，场景步长
        self.N_STEPS = 1

        # 每次输出plot的轮次
        self.polt_num = 50000
        self.save_plot_path = "/home/cbd109/Users/smy/result/DRL/DQN/"
        self.save_plot_num = 100000

        # PriorityReplay所用
        # alpha值，表示优先级大小对取样概率的重要性
        self.PRIORITY_ALPHA = 0.6
        # beta初始值，表示优先级取样的概率对loss更新的weight
        self.PRIORITY_BETA_START = 0.4
        # 用于获取当前beta值，beta-start->1所用参数，超过此值取1.0
        self.PRIORITY_BETA_FRAMES = 100000
