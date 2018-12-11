import os
import random
import pickle
import torch as t

from utils.data_structures import SumSegmentTree, MinSegmentTree
from utils.hyperparameters import Config

cfg = Config()


class ExperienceReplayMemory:
    def __init__(self, capacity):
        # 回放内存容量
        self.capacity = capacity
        # 存储数组
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        # 超出容量删除队列第一个
        if len(self.memory) > self.capacity:
            del self.memory[0]

    # 根据batch大小随机取样
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size), None, None

    # 保存经验回放内存到文件
    def save_replay(self):
        pickle.dump(self.memory, open(cfg.model_dir + 'saved_agents/exp_replay_agent.dump', 'wb'))

    # 从文件中加载经验回放内存
    def load_replay(self):
        replay_name = cfg.model_dir + 'saved_agents/exp_replay_agent.dump''saved_agents/exp_replay_agent.dump'
        if os.path.isfile(replay_name):
            self.memory = pickle.load(open(replay_name, 'rb'))

    # 重写len方法保持能够返回长度
    def __len__(self):
        return len(self.memory)


# 优先级回放机制
class PrioritizedReplayMemory(ExperienceReplayMemory):
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        初始化继承经验回抽机制
        :param capacity: 经验回收容量
        :param alpha: 优先级随机采样重要性
        :param beta_start: 使用重要性的相关性 0不相关 1完全相关
        :param beta_frames: 计算beta折扣所用，超过此数后beta=1，表示更新权重与优先级取样的损失完全相关
        """
        super(PrioritizedReplayMemory, self).__init__(capacity)
        # # 回放内存容量
        # self.capacity = capacity
        # # 存储数组
        # self.memory = []
        self._next_idx = 0
        assert alpha > 0
        self._alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, transition):
        """
        以循环队列的方式存储经验
        :param transition:
        :return:
        """
        # 下一个id
        idx = self._next_idx
        # 若循环队列未填满则添加，否则替换
        if self._next_idx >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self._next_idx] = transition
        self._next_idx = (self._next_idx + 1) % self.capacity

        # 将新放入的权重设为最大
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    # 返回记忆队列的指定位置的场景
    def _encode_sample(self, indices):
        return [self.memory[i] for i in indices]

    # 以比例方式抽样
    def _sample_proportional(self, batch_size):
        res = []
        # print(len(self.memory), self._it_sum.sum(0, len(self.memory) - 1))
        # print(self._it_sum.sum(0,))
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self.memory) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    # 根据batch大小优先级取样
    def sample(self, batch_size):
        # 回放机制中抽样的位置
        indices = self._sample_proportional(batch_size)
        # 抽样结果的权重
        weights = list()

        # 获取最小权重的概率
        p_min = self._it_min.min() / self._it_sum.sum()
        # 获取beta值，表示权重对取样的重要性
        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        # 获取最大权重
        max_weight = (p_min * len(self.memory)) ** (-beta)

        for idx in indices:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self.memory)) ** (-beta)
            weights.append(weight / max_weight)
        weights = t.tensor(weights, device=cfg.device, dtype=t.float)
        encode_sample = self._encode_sample(indices)

        return encode_sample, indices, weights

    # 更新优先级
    def update_priorities(self, indices, priorities):
        assert len(indices) == len(priorities)
        # 对所有抽样出来的经验更新其优先级
        for idx, priority in zip(indices, priorities):
            assert 0 <= idx < len(self.memory)
            # 优先级为priority**alpha,其中alpha表示优先级重要性
            self._it_sum[idx] = (priority + 1e-5) ** self._alpha
            self._it_min[idx] = (priority + 1e-5) ** self._alpha

            self._max_priority = max(self._max_priority, (priority + 1e-5))

    # 保存经验回放内存到文件
    def save_replay(self):
        pickle.dump(self.memory, open(cfg.model_dir + 'saved_agents/exp_replay_agent.dump', 'wb'))

    # 从文件中加载经验回放内存
    def load_replay(self):
        replay_name = cfg.model_dir + 'saved_agents/exp_replay_agent.dump''saved_agents/exp_replay_agent.dump'
        if os.path.isfile(replay_name):
            self.memory = pickle.load(open(replay_name, 'rb'))

    # 重写len方法保持能够返回长度
    def __len__(self):
        return len(self.memory)
