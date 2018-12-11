import torch as t
import torch.nn as nn
import torch.nn.functional as F

from module.layers import NoisyLinear
from module.network_bodies import SimpleBody

device = t.device("cuda" if t.cuda.is_available() else "cpu")


# DQN
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions, noisy=False, sigma_init=0.5, body=SimpleBody):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy = noisy
        # 根据来源不同选择为MLP——SimpleBody还是CNN——AtariBody
        self.body = body(input_shape, num_actions, noisy, sigma_init)

        self.fc1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(self.body.feature_size(),
                                                                                               512,
                                                                                               sigma_init)
        self.fc2 = nn.Linear(512, self.num_actions) if not self.noisy else NoisyLinear(512, self.num_actions,
                                                                                       sigma_init)

    def forward(self, x):
        x = self.body(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.f1.sample_noise()
            self.f2.sample_noise()


# DQN
class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_actions, noisy=False, sigma_init=0.5, body=SimpleBody):
        super(DuelingDQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy = noisy
        # 根据来源不同选择为MLP——SimpleBody还是CNN——AtariBody
        self.body = body(input_shape, num_actions, noisy, sigma_init)

        self.adv1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(
            self.body.feature_size(),
            512,
            sigma_init)
        self.adv2 = nn.Linear(512, self.num_actions) if not self.noisy else NoisyLinear(512, self.num_actions,
                                                                                        sigma_init)

        self.val1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(
            self.body.feature_size(),
            512,
            sigma_init)

        self.val2 = nn.Linear(512, 1) if not self.noisy else NoisyLinear(512, 1,sigma_init)

    def forward(self, x):
        x = self.body(x)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)

        val = F.relu(self.val1(x))
        val = self.val2(val)

        return val + adv - adv.mean()

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.f1.sample_noise()
            self.f2.sample_noise()
