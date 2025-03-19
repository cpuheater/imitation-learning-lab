import sys

import torch
import gymnasium
from time import sleep
import numpy as np
import torch.nn as nn
import torch
from torch.distributions.categorical import  Categorical
from dataclasses import dataclass
import tyro
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from networks import Simple, ImpalaCNNSmall
import vizdoom as vzd
import cv2
from tqdm.auto import tqdm

@dataclass
class Args:
    env_id: str = "monsters"
    """the id of the environment"""
    num_actions: int = 5
    """num actions"""
    num_episodes: int = 5
    """"""
    model_file: str = "models/monsters_ImpalaCNNLarge_100.pt"
    """monsters_ImpalaCNNSmall_100 monsters_Small_100.pt"""
    skip_frame: int = 2
    """"""
    render: bool = True
    """"""

if __name__ == '__main__':
    args = tyro.cli(Args)
    game = vzd.DoomGame()
    game.load_config(f"scenarios/{args.env_id}.cfg")
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)

    if args.render:
        game.set_window_visible(True)
        game.set_mode(vzd.Mode.ASYNC_PLAYER)
    else:
        game.set_mode(vzd.Mode.PLAYER)

    game.init()
    model = torch.load(args.model_file, weights_only=False, map_location='cpu')
    model.eval()
    total_rewards = []
    for episode in tqdm(range(args.num_episodes)):
        game.new_episode()
        obs = game.get_state().screen_buffer
        while True:
            obs = cv2.resize(obs, (80, 60), interpolation = cv2.INTER_AREA)
            obs = obs.transpose(2,0,1)
            obs = torch.from_numpy(obs).float().unsqueeze(0)
            out = model(obs)[0]
            probs = torch.sigmoid(out)
            max_idx = torch.argmax(probs, 0, keepdim=True)
            one_hot = torch.FloatTensor(probs.shape)
            one_hot.zero_()
            one_hot.scatter_(0, max_idx, 1)
            game.make_action(one_hot.tolist(), args.skip_frame)
            if game.is_episode_finished():
                total_rewards.append(game.get_total_reward())
                print(f"Total reward: {game.get_total_reward()}, running mean: {np.mean(total_rewards)}")
                break
            else:
                obs = game.get_state().screen_buffer
    print(f"Mean reward: {np.mean(total_rewards)} using {args.model_file}")
