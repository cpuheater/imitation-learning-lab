
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Simple(nn.Module):
    def __init__(self, num_channels, num_actions):
        super(Simple, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(num_channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(1536, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, num_actions))
        )

    def forward(self, x):
        x = x / 255
        x = self.network(x)
        return x
