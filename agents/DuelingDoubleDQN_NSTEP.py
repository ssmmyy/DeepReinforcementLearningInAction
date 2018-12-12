import numpy as np
import torch as t
import torch.optim as optim
from agents.BaseAgent import BaseAgent
from module.network_bodies import AtariBody
from module.networks import DuelingDQN
from utils.hyperparameters import Config
from utils.replay_memory import ExperienceReplayMemory

cfg = Config()


class Model(BaseAgent):
    def __init__(self, static_policy=False, env=None, config=None):
        super(Model, self).__init__()
        # 选用训练设备（有GPU则选取GPU:0）
        self.device = config.device
        # 折扣因子
        self.gamma = config.GAMMA
        # 学习率
        self.lr = config.LR
        # 目标函数更新频率
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        # 经验回放内存大小
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        # 批次大小
        self.batch_size = config.BATCH_SIZE
        # 学习开始值
        self.learn_start = config.LEARN_START
        # 是否为静态策略
        self.static_policy = static_policy
        # 特征数（输入维度）
        self.num_feats = env.observation_space.shape
        # action数量（一般为输出维度）
        self.num_actions = env.action_space.n
        # 所用gym环境
        self.env = env
        # 声明网络
        self.declare_networks()
        # 目标网络参数赋值
        self.target_model.load_state_dict(self.model.state_dict())
        # 定义优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 将模型移入指定设备中运行
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        # 若为静态策略则评估，否则训练
        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()
        # 内存值初始化为None，需要declare_memory赋值
        self.memory = None
        # 更新计数器
        self.update_count = 0
        # 声明内存值
        self.declare_memory()

        # 场景步长，DQN_NSTEP模型所用
        self.nsteps = config.N_STEPS
        # 场景状态池，DQN_NSTEP模型所用
        self.nstep_buffer = list()

    # 声明网络模型
    def declare_networks(self):
        # 所用皆为端对端模型，输入维度为环境状态，输出维度为备选action数量
        self.model = DuelingDQN(self.num_feats, self.num_actions, body=AtariBody)
        self.target_model = DuelingDQN(self.num_feats, self.num_actions, body=AtariBody)

    # 声明经验回放机制
    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size)

    # 每nsteps个场景，累计其折扣累加收益放入经验池中
    def append_to_replay(self, s, a, r, s_):
        # 场景buffer添加场景
        self.nstep_buffer.append(((s, a, r, s_)))
        # 若场景小于nsteps个，则跳过
        if len(self.nstep_buffer) < self.nsteps:
            return
        # 计算buffer中的累计折扣收益
        R = sum([self.nstep_buffer[i][2] * self.gamma ** i for i in range(self.nsteps)])
        # 从队列中取出最早的场景
        state, action, _, _ = self.nstep_buffer.pop(0)
        # 将最早的状态，操作以及累计收益和下一个状态放入经验回放机制中
        self.memory.push((state, action, R, s_))

    # 获取经验池中样品，并携带是否为最终状态的标识
    def prep_minibatch(self):
        # 从经验池中随机抽取一个batch的样例
        transitions, indices, weights = self.memory.sample(self.batch_size)
        # 将抽取到的一个batch的transition变成s,a,r,s_四个数组
        # print(transitions.shape)
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)
        # 补充维度
        shape = (-1,) + self.num_feats

        # 将数组转换成计算图张量
        batch_state = t.tensor(batch_state, device=self.device, dtype=t.float).view(shape)
        batch_action = t.tensor(batch_action, device=self.device, dtype=t.long).squeeze().view(-1, 1)
        batch_reward = t.tensor(batch_reward, device=self.device, dtype=t.float).squeeze().view(-1, 1)

        # 处理next state为空的情况
        non_final_mask = t.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device,
                                  dtype=t.uint8)
        # print("non_final_mask ", non_final_mask)
        # print("non_final_mask.shape ", non_final_mask.shape)
        try:
            # 获取到并非完结状态的状态队列
            non_final_next_states = t.tensor([s for s in batch_next_state if s is not None], device=self.device,
                                             dtype=t.float).view(shape)
            # 并非全是完结状态
            empty_next_state_values = False
        except:
            # 全是完结状态
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights

    # 计算loss即预估值与折扣累计值之间差异
    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars
        # 获取指定状态的相应动作神经网络模型输出值
        current_q_values = self.model(batch_state).gather(1, batch_action)

        # 获取期望时固定模型，因此no_grad()
        with t.no_grad():
            # 最大下一个Q值，变成[[0],[0],[0]],shape为(self.batch_size,1)
            max_next_q_values = t.zeros(self.batch_size, device=self.device, dtype=t.float).unsqueeze(dim=1)
            # 若不是所有状态都是完结状态
            if not empty_next_state_values:
                # 获取到下列每一个非完结状态的最优action(Double DQN 所用为model,而非target_model)
                max_next_action = self.get_max_net_state_action(non_final_next_states)
                # 在目标网络非完结状态执行获取到的最优action获取Q值
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)

            # Q(s_i,a_i) = r + gamma * maxQ(s_(i+1),a_(i+1))
            expected_q_values = batch_reward + (self.gamma * max_next_q_values)

        # 获取期望Q值与当前模型得到的Q值之间的差距
        diff = (expected_q_values - current_q_values)
        # 获取huber损失均值
        loss = self.huber(diff)
        loss = loss.mean()

        return loss

    # 更新网络
    def update(self, s, a, r, s_, frame=0):
        # 若是静态模型，则返回空值
        if self.static_policy:
            return None
        # 将状态加入经验回放机制中
        self.append_to_replay(s, a, r, s_)

        # 若小于训练开始值，则返回空值
        if frame < self.learn_start:
            return None
        # 从经验回收机制中获取一个batch的经验
        batch_vars = self.prep_minibatch()
        # 计算损失函数
        loss = self.compute_loss(batch_vars)

        # 初始化当前轮次优化
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.model.parameters():
            # 将参数限制在-1到1之间
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # 更新目标模型
        self.update_target_model()
        # 保存损失与参数量级
        self.save_loss(loss.item())
        self.save_sigma_param_magnitudes()

    # 以epsilon greedy方式给定状态获取最优action
    def get_action(self, s, eps=0.1):
        with t.no_grad():
            # 若大于epsilon或静态模型则执行最优策略
            if np.random.random() >= eps or self.static_policy:
                X = t.tensor([s], device=self.device, dtype=t.float)
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    # 更新模型
    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    # 获取目标网络中最大的reward的index，即对应的action操作
    def get_max_net_state_action(self, next_states):
        # 获取目标网络中最大的reward的index，注：max得到结果共有两维描述，其中[0]为最大值[1]为最大值的index
        return self.model(next_states).max(dim=1)[1].view(-1, 1)

    # 获取huber loss 即差值打于1则为线性损失，小于1则为平方损失
    def huber(self, x):
        cond = (x.abs() < 1.0).to(t.float)
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)

    # nstep结束
    def finish_nstep(self):
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2] * self.gamma ** i for i in range(len(self.nstep_buffer))])
            state, action, _, _ = self.nstep_buffer.pop(0)

            self.memory.push((state, action, R, None))
