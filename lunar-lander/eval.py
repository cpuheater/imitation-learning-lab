import sys

import torch
import gymnasium
from time import sleep
import numpy as np
import torch.nn as nn
import torch
from torch.distributions.categorical import  Categorical
from model import Agent

def main():

    env = gymnasium.make("LunarLander-v2", render_mode="human")
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    agent = Agent(env)
    agent.load_state_dict(torch.load("./models/agent.pt"))
    agent.eval()

    obs, _ = env.reset()
    done = False
    for i in range(10000):
        env.render()
        obs = torch.from_numpy(obs).float()
        action, _, _ = agent.get_action(obs)
        obs, rewards, terminations, truncations, info = env.step(action.cpu().numpy())
        done = np.logical_or(terminations, truncations)
        sleep(0.001)
        if done:
            obs, _ = env.reset()


if __name__ == '__main__':
    sys.exit(main())