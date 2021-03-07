# https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/core/environment.py

import numpy as np
from collections import deque
import cv2
cv2.ocl.setUseOpenCL(False)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import Agent, MicroRTSStatsRecorder, VecMonitor, VecPyTorch
import pickle
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


envs = MicroRTSVecEnv(
    num_envs=1,
    render_theme=2,
    ai2s=[microrts_ai.workerRushAI for _ in range(1)],
    map_path="maps/10x10/basesWorkers10x10.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)
envs = MicroRTSStatsRecorder(envs)
envs = VecMonitor(envs)
envs = VecPyTorch(envs, device)

def enjoy(agent):
    rewards = []
    next_obs = envs.reset()
    while True:
        action, logproba, _, _ = agent.get_action(next_obs, envs=envs)
        try:
            next_obs, rs, ds, infos = envs.step(action.T)
            rewards.append(rs)
        except Exception as e:
            e.printStackTrace()
            raise
        for idx, info in enumerate(infos):
            if 'episode' in info.keys():
                print(info['microrts_stats']['WinLossRewardFunction'])
        if ds:
            break
    return rewards




with open("trajectories.dat", 'rb') as rfp:
    trajectories = pickle.load(rfp)

def get_rollouts(trajectories):
    for t in trajectories:
        yield t[0], t[1], t[2]


def train(agent, x, y, epochs=50):
    x = x.to(device)
    y = y.to(device)
    agent = agent.to(device)
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    accuracy = 0
    optimizer = optim.Adam(agent.parameters(), lr=0.01, eps=1e-5)
    for epoch in range(epochs):
        outputs = agent.forward(x)
        probs = torch.softmax(outputs, dim=1)
        _, predictions = probs.max(1)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        accuracy += torch.sum(predictions == y)

    return running_loss/ y.shape[0], accuracy/y.shape[0]

def train_regularization(agent, x, y, epochs=50):
    x = x.to(device)
    y = y.to(device)
    agent = agent.to(device)
    running_loss = 0.0
    accuracy = 0
    optimizer = optim.Adam(agent.parameters(), lr=0.01, eps=1e-5)
    for epoch in range(epochs):
        _, log_prob, entropy = agent.get_action(x, y)
        log_prob = log_prob.mean()
        entropy = entropy.mean()

        l2_norms = [torch.sum(torch.square(w)) for w in agent.parameters()]
        l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square

        ent_loss = 0.001 * entropy
        neglogp = -log_prob
        l2_loss = 0.0 * l2_norm
        loss = neglogp + ent_loss + l2_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / y.shape[0]

def main():
    data_size = [1, 100, 100, 100, 100, 100, 100]
    for size in data_size:
        x = []
        y = []
        agent = Agent(1, envs).to(device)
        rollouts = get_rollouts(trajectories)
        for states, rewards, actions in rollouts:
            x = torch.from_numpy(states).float()
            y = torch.from_numpy(actions).long()
            train(agent, x, y)
            episode_rewards = []
        for _ in range(50):
            states, actions, rewards = get_rollouts()
            total_reward = np.sum(rewards)
            episode_rewards.append(total_reward)
        print(f'Number of training rollouts: {str(size)}: reward mean: {str(np.mean(episode_rewards))} \n')

if __name__ == "__main__":
    main()


