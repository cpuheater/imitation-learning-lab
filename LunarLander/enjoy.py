import sys

import torch
import gym
from time import sleep
import numpy as np
import torch.nn as nn
import torch
from torch.distributions.categorical import  Categorical
from model import Agent

def main():

    env = gym.make("LunarLander-v2")
    agent = Agent(env)
    agent.load_state_dict(torch.load("./models/agent.pt"))
    agent.eval()

    obs = env.reset()
    done = False
    for i in range(10000):
        env.render()
        obs = torch.from_numpy(obs).float()
        action, _, _ = agent.get_action(obs)
        obs, rew, done, info = env.step(action.cpu().numpy())
        sleep(0.001)
        if done:
            obs = env.reset()


if __name__ == '__main__':
    sys.exit(main())