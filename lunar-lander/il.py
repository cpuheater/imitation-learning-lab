# https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/core/environment.py

import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from model import Agent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = gym.make("LunarLander-v2")
expert = Agent(env)
expert.load_state_dict(torch.load("./models/agent.pt"))
expert.eval()

def generate_rollout(agent, env):
    agent.to("cpu")
    states = []
    actions = []
    rewards = []
    obs = env.reset()
    steps = 0
    while True:
        obs = torch.from_numpy(obs).float()
        states.append(obs)
        logits = agent.forward(obs)
        probs = torch.softmax(logits, dim=0)
        action = probs.argmax()
        obs, reward, done, info = env.step(action.cpu().numpy())
        actions.append(action.long().view(1))
        rewards.append(reward)
        steps += 1
        if done:
            break
    return states, actions, rewards

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
        student = Agent(env)
        for episode in range(size):
            states, actions, rewards = generate_rollout(expert, env)
            x.extend(states)
            y.extend(actions)
        x = torch.stack(x)
        y = torch.cat(y)
        train(student, x, y)
        episode_rewards = []
        for _ in range(50):
            states, actions, rewards = generate_rollout(student, env)
            total_reward = np.sum(rewards)
            episode_rewards.append(total_reward)
        print(f'Number of training rollouts: {str(size)}: reward mean: {str(np.mean(episode_rewards))} \n')

if __name__ == "__main__":
    main()


