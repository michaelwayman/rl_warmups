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
from rl_warmups.common.params import BaseParams
from rl_warmups.common.tensorboard import Tensorboard


@dataclass
class Params(BaseParams):
    env_id: str = "CartPole"
    algo: str = "reinforce"

    learning_rate: float = 0.01
    gamma: float = 0.99


class PolicyNetwork(torch.nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()
        obs_size = np.array(env.observation_space.shape).prod()
        action_size = env.action_space.n
        self.fc1 = torch.nn.Linear(in_features=obs_size, out_features=128)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=action_size)

    def forward(self, obs):
        x = self.fc1(obs)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=-1)


class ReinforceAgent(BaseAgent):
    def __init__(self, params: Params, env: gym.Env):
        super().__init__(params, env)
        self.network = PolicyNetwork(env)
        self.optimizer = torch.optim.Adam(
            params=self.network.parameters(),
            lr=params.learning_rate,
        )

    def get_action(self, obs: torch.Tensor):
        with torch.no_grad():
            action = self.network(obs)
        action = torch.distributions.Categorical(action).sample().item()
        return action

    def update_network(self, episode=None, **kwargs):
        with torch.no_grad():
            observations = torch.stack([e[0] for e in episode], dim=0)
            actions = torch.tensor([e[1] for e in episode], dtype=torch.int32)
            rewards = [e[3] for e in episode]
            rewards = torch.tensor(utils.calc_discounted_returns(self.params.gamma, rewards), dtype=torch.float32)
            rewards = utils.z_score(rewards)

        action_probs = self.network(observations)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)

        loss = -(log_probs * rewards).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def main(params: Params, trial: Optional[optuna.Trial] = None):
    tensorboard = Tensorboard(params)
    tensorboard.record_params()

    torch.manual_seed(params.seed)

    env = utils.make_env(params)

    agent = ReinforceAgent(params, env)
    final_running_reward = utils.train_agent__mc(agent, tensorboard, trial)
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
    args = parser.parse_args()

    hparams = Params(**{k: v for k, v in vars(args).items() if v})
    main(hparams)
    # utils.evaluate(Agent, hparams)
