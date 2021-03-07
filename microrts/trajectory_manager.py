import collections
import numpy as np
import pickle
import os

Data = collections.namedtuple('Data', ['obs', 'action', 'reward'])

class TrajectoryManager(object):

    def __init__(self, num_envs, file_name, envs):
        self.dir = dir
        self.file_name = file_name
        self.num_envs = num_envs
        self.trajectories = []
        self.envs = envs
        for _ in range(num_envs):
            self.trajectories.append([np.empty(shape=(0,) + envs.observation_space.shape),
                                             np.empty(0,), np.empty(shape=(0, )+envs.action_space.shape)])
        self.num_episodes = 0
        if os.path.exists(file_name):
            with open(file_name, 'rb') as rfp:
                self.successful_trajectories = pickle.load(rfp)
        else:
            self.successful_trajectories = []

    def process(self, b_rewards, b_obs, b_actions, b_dones, next_done, next_obs, b_wins):
        steps = b_rewards.shape[0]
        b_rewards = b_rewards.transpose(1, 0)
        b_dones = b_dones.transpose(1, 0)[:, 1:]
        b_dones = np.append(b_dones, next_done.reshape(-1, 1), axis=1)
        b_obs = b_obs.transpose(1, 0, 2, 3, 4)
        b_actions = b_actions.transpose(1, 0, 2)
        b_wins = b_wins.transpose(1, 0)

        for i in range(self.num_envs):
            for j in range(steps):
                self.trajectories[i][0] = np.vstack((self.trajectories[i][0], np.expand_dims(b_obs[i][j], axis=0)))
                self.trajectories[i][1] = np.append(self.trajectories[i][1], b_rewards[i][j])
                self.trajectories[i][2] = np.vstack((self.trajectories[i][2], np.expand_dims(b_actions[i][j], axis=0)))
                if b_wins[i][j]:
                    self.successful_trajectories.append(self.trajectories[i].copy())
                    print(f"step {j} dones: {b_dones[i][j]} reward: {b_rewards[i][j]}")
                    with open(self.file_name, 'wb') as wfp:
                        pickle.dump(self.successful_trajectories, wfp)
                    self.trajectories[i] = [np.empty(shape=(0,) + self.envs.observation_space.shape),
                                             np.empty(0,), np.empty(shape=(0, ) + self.envs.action_space.shape)]
                if b_dones[i][j]:
                    self.trajectories[i] = [np.empty(shape=(0,) + self.envs.observation_space.shape),
                                             np.empty(0,), np.empty(shape=(0, ) + self.envs.action_space.shape)]

    def get(self):
        return iter(self.successful_trajectories)

