import numpy as np
import torch.nn as nn
import torch
from torch.distributions.categorical import  Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 128)),
            nn.ReLU()
        )
        self.actor = layer_init(nn.Linear(128, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def forward(self, x):
        return self.actor(self.network(x))

    def get_action(self, x, action=None):
        logits = self.forward(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_value(self, x):
        return self.critic(self.network(x))