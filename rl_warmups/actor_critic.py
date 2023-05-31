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
    env_id: str = "CartPole-v1"
    algo: str = "actor_critic"

    learning_rate: float = 0.01
    gamma: float = 0.99
    seed: int = 1


class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()
        obs_size = np.array(env.observation_space.shape).prod()
        action_size = env.action_space.n
        self.fc1 = torch.nn.Linear(in_features=obs_size, out_features=128)
        self.a1 = torch.nn.Linear(in_features=128, out_features=action_size)
        self.c1 = torch.nn.Linear(in_features=128, out_features=1)

    def forward(self, obs):
        x = self.fc1(obs)
        x = F.relu(x)
        a = self.a1(x)
        c = self.c1(x)
        return F.softmax(a, dim=-1), c


class ActorCriticAgent(BaseAgent):
    def __init__(self, params: Params, env: gym.Env):
        super().__init__(params, env)
        self.network = ActorCriticNetwork(env)
        self.optimizer = torch.optim.Adam(
            params=self.network.parameters(),
            lr=params.learning_rate,
        )

    def get_action(self, obs: torch.Tensor):
        with torch.no_grad():
            action, _ = self.network(obs)
        action = torch.distributions.Categorical(action).sample().item()
        return action

    def update_network(self, episode=None, **kwargs):
        with torch.no_grad():
            observations = torch.stack([e[0] for e in episode], dim=0)
            actions = torch.tensor([e[1] for e in episode], dtype=torch.int32)
            rewards = [e[3] for e in episode]
            rewards = torch.tensor(utils.calc_discounted_returns(self.params.gamma, rewards), dtype=torch.float32)
            rewards = utils.z_score(rewards)

        action_probs, values = self.network(observations)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        values = values.flatten()

        with torch.no_grad():
            advantages = rewards - values
        policy_loss = -(log_probs * advantages).sum()
        critic_loss = F.smooth_l1_loss(values, rewards, reduction="sum")

        self.optimizer.zero_grad()
        loss = policy_loss + critic_loss
        loss.backward()
        self.optimizer.step()


def main(params: Params, trial: Optional[optuna.Trial] = None):
    tensorboard = Tensorboard(params)
    tensorboard.record_params()

    torch.manual_seed(params.seed)

    env = utils.make_env(params)

    agent = ActorCriticAgent(params, env)
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
