# https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/core/environment.py

import numpy as np
from collections import deque
import gymnasium
import cv2
cv2.ocl.setUseOpenCL(False)
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import Agent
from time import sleep
import time
import os
from torch.utils.tensorboard import SummaryWriter
import torch
import random
import tyro
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

@dataclass
class Args:
    env_id: str = "LunarLander-v2"
    """the id of the environment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """`torch.backends.cudnn.deterministic=False`"""
    lr: float = 0.01
    """the learning rate of the optimizer"""
    rollout_steps: int = 20000
    """"""
    epochs: int = 50
    """num train epochs"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):
    def __init__(self, inputs, labels, transform=None):
        self.inputs = inputs
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.labels[index]

        #if self.transform:
            #x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
        #    x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.inputs)

def generate_rollout(agent, env, num_steps=None):
    agent.to("cpu")
    states = []
    actions = []
    rewards = []
    obs, _ = env.reset()
    steps = 0
    while True:
        obs = torch.from_numpy(obs).float()
        states.append(obs)
        logits = agent.forward(obs)
        probs = torch.softmax(logits, dim=0)
        action = probs.argmax()
        obs, reward, terminations, truncations, info = env.step(action.cpu().numpy())
        done = np.logical_or(terminations, truncations)
        actions.append(action.long().view(1))
        rewards.append(reward)
        steps += 1
        if (num_steps and num_steps <= steps) or (done and not num_steps):
            break
        if done:
            obs, _ = env.reset()
    return states, actions, rewards

def train(agent, x, y, epochs=50):
    x = x.to(device)
    y = y.to(device)
    agent = agent.to(device)
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    accuracy = 0
    optimizer = optim.Adam(agent.parameters(), lr=0.01, eps=1e-5)
    dataset = MyDataset(data = x, targets = y)
    dataloader = DataLoader(dataset, batch_size=5)
    for epoch in range(epochs):
        for (b_input, b_labels) in dataloader:
            outputs = agent.forward(b_input)
            probs = torch.softmax(outputs, dim=1)
            _, predictions = probs.max(1)
            loss = criterion(outputs, b_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            accuracy += torch.sum(predictions == y)

    return running_loss/ y.shape[0], accuracy/y.shape[0]

def train_regularization(writer, agent, inputs, labels, epochs=50, lr=0.01):
    inputs = inputs.to(device)
    labels = labels.to(device)
    agent = agent.to(device)
    running_loss = 0.0
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    dataset = MyDataset(inputs = inputs, labels = labels)
    dataloader = DataLoader(dataset, batch_size=250)
    running_loss_history = []
    for epoch in range(epochs):
        running_loss = 0.0
        for (input, labels) in dataloader:
            _, log_prob, entropy = agent.get_action(input, labels)
            log_prob = log_prob.mean()
            entropy = entropy.mean()

            l2_norms = [torch.sum(torch.square(w)) for w in agent.parameters()]
            l2_norm = sum(l2_norms) / 2

            ent_loss = 0.001 * entropy
            neglogp = -log_prob
            l2_loss = 0.0 * l2_norm
            loss = neglogp + ent_loss + l2_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss = running_loss/len(dataloader)
        running_loss_history.append(running_loss)
        writer.add_scalar("losses/reg_loss", running_loss, epoch)

    return np.mean(running_loss_history)

if __name__ == "__main__":

    args = tyro.cli(Args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    exp_name = os.path.basename(__file__)[: -len(".py")]
    run_name = f"{exp_name}__{args.env_id}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    env = gymnasium.make(args.env_id)
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    expert = Agent(env)
    expert.load_state_dict(torch.load("./models/agent.pt"))
    expert.eval()
    student = Agent(env)
    states, actions, rewards = generate_rollout(expert, env, args.rollout_steps)
    inputs = torch.stack(states)
    labels = torch.cat(actions)
    train_regularization(writer, student, inputs, labels, args.epochs, args.lr)
    episode_rewards = []
    for _ in range(50):
        env = gymnasium.make("LunarLander-v2", render_mode="human")
        states, actions, rewards = generate_rollout(student, env)
        total_reward = np.sum(rewards)
        episode_rewards.append(total_reward)
    print(f'Number of training rollouts: {str(len(states))}: reward mean: {str(np.mean(episode_rewards))} \n')



