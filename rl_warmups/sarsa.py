import argparse
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
    env_id: str = "LunarLander-v2"

    buffer_size: int = 500000
    batch_size: int = 16
    num_replays: int = 4

    learning_rate: float = 0.001
    gamma: float = 0.99
    tau: float = 0.9


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


class SarsaAgent(BaseAgent):
    def __init__(self, params: Params, env: gym.Env):
        super().__init__(params, env)
        self.replay_buffer = ReplayBuffer(
            params.buffer_size,
            env.observation_space.sample(),
            env.action_space.sample(),
        )
        self.network = QNetwork(params, env)
        self.optimizer = torch.optim.Adam(params=self.network.parameters())
        self.target_network = QNetwork(params, env)
        self.target_network.requires_grad_(False)
        self.target_network.load_state_dict(self.network.state_dict())

    def get_action(self, obs: torch.Tensor):
        with torch.no_grad():
            q_values = self.network(obs)
        probs = F.softmax(q_values / self.params.tau, dim=-1)
        action = torch.distributions.Categorical(probs=probs).sample().item()
        return action

    def update_network(self, **kwargs):
        if len(self.replay_buffer) < self.params.batch_size:
            return

        for _ in range(self.params.num_replays):
            batch = self.replay_buffer.sample(self.params.batch_size)
            with torch.no_grad():
                q_next_mat = self.target_network.forward(batch.next_obs)
                v_next_vec = (q_next_mat * F.softmax(q_next_mat / self.params.tau, dim=-1)).sum(dim=-1) * (
                    1 - batch.dones
                )
                target_vec = batch.rewards + self.params.gamma * v_next_vec
            q_mat = self.network.forward(batch.obs)
            q_vec = q_mat.gather(1, batch.actions.reshape(-1, 1)).flatten()
            loss = F.smooth_l1_loss(q_vec, target_vec)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.target_network.load_state_dict(self.network.state_dict())


def main(params: Params, trial: Optional[optuna.Trial] = None):
    tensorboard = Tensorboard(params)
    tensorboard.record_params()

    torch.manual_seed(params.seed)

    env = utils.make_env(params)

    agent = SarsaAgent(params, env)
    final_running_reward = utils.train_agent__td(agent, tensorboard, trial)
    env.close()
    return np.array(final_running_reward).mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--env-id", type=str)
    parser.add_argument("--render", type=bool)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--buffer-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-replays", type=int)
    args = parser.parse_args()
    hparams = Params(**{k: v for k, v in vars(args).items() if v})
    main(hparams)
