from typing import NamedTuple

import gymnasium as gym
import numpy as np
import torch


class ReplayBufferSample(NamedTuple):
    obs: torch.Tensor
    actions: torch.Tensor
    next_obs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, size: int, obs_like, action_like):
        self.size = size
        assert len(obs_like.shape) > 0
        self.observations = np.zeros(tuple([self.size]) + obs_like.shape, dtype=obs_like.dtype)
        self.actions = np.zeros(tuple([self.size]) + action_like.shape, dtype=action_like.dtype)
        self.next_observations = np.zeros_like(self.observations)
        self.rewards = np.zeros(tuple([self.size]), dtype=np.float32)
        self.dones = np.zeros(tuple([self.size]), dtype=np.float32)
        self.pos = 0
        self.full = False

    def __len__(self):
        return self.size if self.full else self.pos

    @classmethod
    def from_env(cls, size: int, env: gym.Env):
        return cls(
            size=size,
            obs_like=env.observation_space.sample(),
            action_like=env.action_space.sample(),
        )

    def add(self, obs, action, next_obs, reward, done):
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.next_observations[self.pos] = next_obs
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSample:
        upper_bound = self.size if self.full else self.pos
        batch_idx = np.random.randint(0, upper_bound, size=batch_size)
        actions = self.actions.reshape(-1, 1) if len(self.actions.shape) == 0 else self.actions
        return ReplayBufferSample(
            torch.from_numpy(self.observations[batch_idx]),
            torch.from_numpy(actions[batch_idx]),
            torch.from_numpy(self.next_observations[batch_idx]),
            torch.from_numpy(self.rewards[batch_idx]),
            torch.from_numpy(self.dones[batch_idx]),
        )
