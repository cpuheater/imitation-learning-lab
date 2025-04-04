import argparse
import os
import matplotlib.pyplot as plt
import torch
import cv2
from argparse import ArgumentParser
import os
from dataclasses import dataclass
from time import sleep
import vizdoom as vzd
import tyro
import numpy as np
import time
import keyboard
import vizdoom as vzd

@dataclass
class Args:
    env_id: str = "basic"
    """the id of the environment"""
    seed: int = 1
    """seed of the experiment"""
    episodes: int = 50
    """num of episodes"""
    data_dir: str = "data"
    """"""
    skip_frames: str = 2


if __name__ == '__main__':
    args = tyro.cli(Args)

    isExist = os.path.exists(args.data_dir)
    if not isExist:
        os.makedirs(args.data_dir)

    game = vzd.DoomGame()

    game.load_config(f"scenarios/{args.env_id}.cfg")
    #game.add_game_args("+freelook 1")
    game.add_game_args('+"bind w +forward" +"bind s +back" +"bind a +left" +"bind d +right"')
    game.set_mode(vzd.Mode.SPECTATOR)
    game.set_window_visible(True)
    game.init()
    obs = []
    actions = []
    total_frames = 0
    buttons = game.get_available_buttons()
    for episode in range(args.episodes):
        print("Episode #" + str(episode + 1))
        game.new_episode()
        frames = 0
        while not game.is_episode_finished():
            state = game.get_state()
            game.advance_action(args.skip_frames)
            img = state.screen_buffer
            last_action = game.get_last_action()
            reward = game.get_last_reward()
            # img (HWC)
            img = cv2.resize(img, (80, 60), interpolation = cv2.INTER_AREA)
            img = img.transpose(2,0,1) # (HWC) -> CHW
            obs.append(torch.from_numpy(img))
            actions.append(torch.tensor(last_action, dtype=torch.float32))
            print("Frames " + str(frames))
            print("Game variables: ", state.game_variables)
            print("Action:", last_action)
            print("Reward:", reward)
            print("Episode:", str(episode + 1))
            print("=====================")
            frames += 1
        total_frames += frames
        print("Total reward:", game.get_total_reward())
    game.close()
    actions = torch.stack(actions)
    obs = torch.stack(obs)
    torch.save((obs, actions),f"{args.data_dir}/{args.env_id}_{total_frames}_{time.strftime('%Y_%m_%d__%H_%M_%S')}.pt")
    print(f"Model saved ! Total frames: {total_frames}")