import torch as t
import torch.nn as nn
import torch.nn.functional as F
import math


# 有偏置项的因子分解式噪声层
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4, factorised_noise=True):
        super(NoisyLinear, self).__init__()
        # 输入，输出维度
        self.in_features = in_features
        self.out_features = out_features
        # 初始化标准差
        self.std_init = std_init
        # 因子分解参数
        self.factorised_noise = factorised_noise
        # mu，sigma权重
        self.weight_mu = nn.Parameter(t.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(t.empty(out_features, in_features))
        # 权重epsilon
        self.register_buffer('weight_epsilon', t.empty(out_features, in_features))
        # mu，sigma的偏置
        self.bias_mu = nn.Parameter(t.empty(out_features))
        self.bias_sigma = nn.Parameter(t.empty(out_features))
        # 偏置eposilon
        self.register_buffer('bias_epsilon', t.empty(out_features))

        self.reset_parameters()
        self.sample_noise()

    # 重新设置参数
    def reset_parameters(self):
        # 获取到mu的范围： 1/输入维度的平方根
        mu_range = 1.0 / math.sqrt(self.in_features)
        # 在指定范围内随机初始化mu
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        # 将sigma全部设为标准差/输入维度的平方根
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        # 偏置值与权重值随机设置方式相同
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

    # 获取带符号的平方根
    def _scale_noise(self, size):
        x = t.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    # 噪声取样
    def sample_noise(self):
        # 以指定因子分解方式生成噪音，即输入输出单独生成噪音向量，并指定相乘
        if self.factorised_noise:
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
            self.bias_epsilon.copy_(epsilon_out)
        # 随机噪音
        else:
            self.weight_epsilon.copy_(t.randn(self.out_features, self.in_features))
            self.bias_epsilon.copy_(t.randn(self.out_features))

    # 前向传播
    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)
