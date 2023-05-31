from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import optuna
import torch
from torch.functional import F

from rl_warmups.common import utils
from rl_warmups.common.agents import BaseAgent
from rl_warmups.common.buffers import ReplayBuffer
from rl_warmups.common.params import BaseParams
from rl_warmups.common.tensorboard import Tensorboard


@dataclass
class Params(BaseParams):
    env_id: str = "CartPole"

    n_random_steps: int = 0
    n_steps = 20000

    buffer_size: int = 100000
    batch_size: int = 32
    num_replays: int = 4

    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 0.5
    end_epsilon: float = 0.0
    eps_fraction: float = 0.5

    def get_epsilon(self, current_step):
        frac_steps = self.n_steps * self.eps_fraction
        eps = (self.epsilon - self.end_epsilon) * (frac_steps - current_step) / frac_steps
        return max(self.end_epsilon, eps)


class QNetwork(torch.nn.Module):
    def __init__(self, params: Params, env: gym.Env):
        super().__init__()
        self.params = params
        obs_size = np.array(env.observation_space.shape).prod()
        action_size = env.action_space.n
        self.fc1 = torch.nn.Linear(in_features=obs_size, out_features=256)
        self.fc2 = torch.nn.Linear(in_features=256, out_features=action_size)

    def forward(self, obs):
        x = self.fc1(obs)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class DQNAgent(BaseAgent):
    def __init__(self, params: Params, env: gym.Env):
        super().__init__(params, env)
        self.replay_buffer = ReplayBuffer(
            self.params.buffer_size,
            self.env.observation_space.sample(),
            self.env.action_space.sample(),
        )
        self.network = QNetwork(params, env)
        self.optimizer = torch.optim.Adam(params=self.network.parameters())
        self.target_network = QNetwork(params, env)
        self.target_network.requires_grad_(False)
        self.target_network.load_state_dict(self.network.state_dict())
        self.global_step = 0

    def get_action(self, obs: torch.Tensor):
        if (
            np.random.rand() > 1 - self.params.get_epsilon(self.global_step)
            or self.global_step < self.params.n_random_steps
        ):
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                logits = self.network(obs)
                action = torch.argmax(logits).item()
        return action

    def update_network(self, **kwargs):
        if len(self.replay_buffer) < self.params.batch_size:
            return

        for _ in range(self.params.num_replays):
            batch = self.replay_buffer.sample(self.params.batch_size)
            with torch.no_grad():
                q_next_mat = self.target_network.forward(batch.next_obs)
                v_next_vec = q_next_mat.max(dim=1).values * (1 - batch.dones)
                target_vec = batch.rewards + self.params.gamma * v_next_vec
            q_mat = self.network.forward(batch.obs)
            q_vec = q_mat.gather(1, batch.actions.reshape(-1, 1)).flatten()
            loss = F.mse_loss(q_vec, target_vec)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.target_network.load_state_dict(self.network.state_dict())


def main(params, trial: Optional[optuna.Trial] = None):
    tensorboard = Tensorboard(params)
    tensorboard.record_params()

    torch.manual_seed(params.seed)

    env = utils.make_env(params)
    agent = DQNAgent(params, env)

    final_running_reward = utils.train_agent__td(agent, tensorboard, trial)
    env.close()
    return np.array(final_running_reward).mean()


if __name__ == "__main__":
    hparams = Params()
    main(hparams)
