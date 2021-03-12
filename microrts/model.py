import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import numpy as np
import gym
import gym_microrts
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
import time
import random
import os
from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder

device = torch.device('cpu')




def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# ALGO LOGIC: initialize agent here:
class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)


class Agent(nn.Module):
    def __init__(self, num_envs, envs):
        super(Agent, self).__init__()
        self.num_envs = num_envs
        self.envs = envs
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 10, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(10, 20, kernel_size=2)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(180, 180)),
            nn.ReLU(), )
        self.actor = layer_init(nn.Linear(180, envs.action_space.nvec.sum()), std=0.01)
        self.critic = layer_init(nn.Linear(180, 1), std=1)

    def forward(self, x):
        logits = self.actor(self.network(x.permute((0, 3, 1, 2))))
        split_logits = torch.split(logits, self.envs.action_space.nvec.tolist(), dim=1)
        return split_logits

    def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
        split_logits = self.forward(x)

        if action is None:
            # 1. select source unit based on source unit mask
            source_unit_mask = torch.Tensor(np.array(envs.vec_client.getUnitLocationMasks()).reshape(self.num_envs, -1))
            multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=source_unit_mask)]
            action_components = [multi_categoricals[0].sample()]
            # 2. select action type and parameter section based on the
            #    source-unit mask of action type and parameters
            # print(np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(args.num_envs, -1))
            source_unit_action_mask = torch.Tensor(
                np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(self.num_envs,
                                                                                                         -1))
            split_suam = torch.split(source_unit_action_mask, envs.action_space.nvec.tolist()[1:], dim=1)
            multi_categoricals = multi_categoricals + [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                                                       zip(split_logits[1:], split_suam)]
            invalid_action_masks = torch.cat((source_unit_mask, source_unit_action_mask), 1)
            action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
            action = torch.stack(action_components)
        else:
            split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_space.nvec.tolist(), dim=1)
            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                                  zip(split_logits, split_invalid_action_masks)]
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action, logprob.sum(0), entropy.sum(0), invalid_action_masks

    def get_value(self, x):
        return self.critic(self.network(x.permute((0, 3, 1, 2))))

class VecMonitor(VecEnvWrapper):
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1

        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                ret = self.eprets[i]
                eplen = self.eplens[i]
                epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)}
                info['episode'] = epinfo
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
                newinfos[i] = info
        return obs, rews, dones, newinfos

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

class MicroRTSStatsRecorder(VecEnvWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        obs = self.venv.reset()
        self.raw_rewards = [[] for _ in range(self.num_envs)]
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        for i in range(len(dones)):
            self.raw_rewards[i] += [infos[i]["raw_rewards"]]
        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                raw_rewards = np.array(self.raw_rewards[i]).sum(0)
                raw_names = [str(rf) for rf in self.rfs]
                info['microrts_stats'] = dict(zip(raw_names, raw_rewards))
                self.raw_rewards[i] = []
                newinfos[i] = info
        return obs, rews, dones, newinfos

