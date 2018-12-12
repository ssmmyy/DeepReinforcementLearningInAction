import numpy as np
import pickle
import os.path
import torch as t
from utils.hyperparameters import Config

cfg = Config()

class BaseAgent(object):
    def __init__(self):
        # 基础模型
        self.model = None
        # 目标模型
        self.target_model = None
        # 优化器
        self.optimizer = None
        # 损失
        self.losses = []
        # 收益
        self.rewards = []
        # 参数量级
        self.sigma_parameter_mag = []

    # huber损失函数
    def huber(self, x):
        """
        huber loss 定义theta值若大于theta值则为线性损失，小于theta值则为平方损失
        :param x:
        :return:
        """
        # 满足条件则为1.0，否则为0.0
        cond = (x.abs() < 1.0).float().detach()
        # cond为1时是平方，否则为线性损失
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

    # 保存模型权重
    def save_weight(self, model_name=""):
        t.save(self.model.state_dict(), cfg.model_dir + 'model' + model_name + '.dump')
        t.save(self.optimizer.state_dict(), cfg.model_dir + 'optimizer' + model_name + '.dump')

    # 保存经验回放内存到文件
    def save_replay(self, model_name=""):
        pickle.dump(self.memory, open(cfg.model_dir + 'saved_agents/exp_replay_agent' + model_name + '.dump', 'wb'))

    # 从文件中加载经验回放内存
    def load_replay(self, model_name=""):
        replay_name = cfg.model_dir + 'saved_agents/exp_replay_agent.dump''saved_agents/exp_replay_agent' + model_name + '.dump'
        if os.path.isfile(replay_name):
            self.memory = pickle.load(open(replay_name, 'rb'))

    # 加载模型权重
    def load_weight(self):
        model_path = cfg.model_dir + 'model.dump'
        optimizer_path = cfg.model_dir + 'optimizer.dump'

        if os.path.isfile(model_path):
            self.model.load_state_dict(t.load(model_path))
        if os.path.isfile(optimizer_path):
            self.optimizer.load_state_dict(t.load(optimizer_path))

    # 保存参数量级
    def save_sigma_param_magnitudes(self):
        tmp = []
        # 遍历所有参数
        for name, param in self.model.named_parameters():
            # 获取需要更新的参数
            if param.requires_grad:
                # 若参数中存在sigma则追加到tmp中
                if "sigma" in name:
                    tmp += param.data.cpu().numpy().ravel().tolist()

        # 若获得参数不为空，则将其均值加入量级中
        if tmp:
            self.sigma_parameter_mag.append(np.mean(np.abs(np.array(tmp))))

    # 保存损失
    def save_loss(self, loss):
        self.losses.append(loss)

    # 保存收获
    def save_reward(self, reward):
        self.rewards.append(reward)
