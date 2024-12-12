
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

class ImpalaCNNSmall(nn.Module):
    """
    Implementation of the small variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, channels, actions):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.pool = torch.nn.AdaptiveMaxPool2d((6, 6))

        self.out = nn.Sequential(nn.Linear(1152, 256),
                              nn.ReLU(),
                              nn.Linear(256, actions))

    def forward(self, x):
        x = x / 255
        f = self.main(x)
        f = self.pool(f)
        f = torch.flatten(f, start_dim=1)
        return self.out(f)


class ImpalaCNNResidual(nn.Module):
    """
    Simple residual block used in the large IMPALA CNN.
    """
    def __init__(self, depth):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv_0 = nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_ = self.conv_0(self.relu(x))
        x_ = self.conv_1(self.relu(x_))
        return x+x_

class ImpalaCNNBlock(nn.Module):
    """
    Three of these blocks are used in the large IMPALA CNN.
    """
    def __init__(self, channels, channels_out):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels_out, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)
        self.residual_0 = ImpalaCNNResidual(channels_out)
        self.residual_1 = ImpalaCNNResidual(channels_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.residual_0(x)
        x = self.residual_1(x)
        return x


class ImpalaCNNLarge(nn.Module):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, channels, actions, model_size=1):
        super().__init__()

        self.main = nn.Sequential(
            ImpalaCNNBlock(channels, 16*model_size),
            ImpalaCNNBlock(16*model_size, 32*model_size),
            ImpalaCNNBlock(32*model_size, 32*model_size),
            nn.ReLU()
        )

        self.pool = torch.nn.AdaptiveMaxPool2d((8, 8))

        self.out = nn.Sequential(nn.Linear(2048*model_size, 256),
                          nn.ReLU(),
                          nn.Linear(256, actions))


    def forward(self, x):
        x = x / 255
        f = self.main(x)
        f = self.pool(f)
        f = torch.flatten(f, start_dim=1)
        return self.out(f)


def get_network(name, channels, num_actions) -> nn.Module:
    if name == "Simple":
        model = Simple(channels, num_actions)
    elif name == "ImpalaCNNSmall":
        model = ImpalaCNNSmall(channels, num_actions)
    elif name == "ImpalaCNNLarge":
        model = ImpalaCNNLarge(channels, num_actions)
    return model




