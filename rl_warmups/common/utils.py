from collections import deque
from typing import Type

import gymnasium as gym
import numpy as np
import optuna
import torch

from rl_warmups.common.agents import BaseAgent
from rl_warmups.common.params import BaseParams

eps = np.finfo(np.float32).eps.item()


def make_env(params: BaseParams):
    kwargs = {}
    if params.render:
        kwargs["render_mode"] = "human"
    env = gym.make(params.env_id, **kwargs)
    env = gym.wrappers.TransformObservation(env, lambda x: torch.from_numpy(x))
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.observation_space.seed(params.seed)
    env.action_space.seed(params.seed)
    env.reset(seed=params.seed)
    return env


def iter_episodes__monte_carlo(agent):
    obs, _ = agent.env.reset()
    episode = []
    for step in range(agent.params.n_steps):
        action = agent.get_action(obs.reshape(1, -1))
        next_obs, reward, terminated, truncated, info = agent.env.step(action)
        episode.append((obs, action, next_obs, reward, terminated))
        if terminated or truncated:
            yield episode, info, step
            episode = []
            next_obs, _ = agent.env.reset()
        obs = next_obs


def iter_episodes__td(agent):
    obs, _ = agent.env.reset()
    for step in range(agent.params.n_steps):
        action = agent.get_action(obs.reshape(1, -1))
        next_obs, reward, terminated, truncated, info = agent.env.step(action)
        agent.replay_buffer.add(obs, action, next_obs, reward, terminated)
        yield info, step
        if terminated or truncated:
            next_obs, _ = agent.env.reset()
        obs = next_obs


def calc_discounted_returns(gamma: float, rewards: list):
    discounted_returns = []
    prev_reward = 0
    for reward in reversed(rewards):
        discounted_reward = reward + gamma * prev_reward
        prev_reward = discounted_reward
        discounted_returns.append(discounted_reward)
    return discounted_returns[::-1]


def z_score(t: torch.Tensor):
    return (t - t.mean()) / (t.std() + eps)


def evaluate(Agent: Type[BaseAgent], params: BaseParams):
    params.render = True
    env = make_env(params)
    agent = Agent(params=params, env=env)
    agent.load_weights()
    while True:
        obs, info = agent.env.reset()
        done = False
        while not done:
            action = agent.get_action(obs.reshape(1, -1))
            obs, reward, terminated, truncated, info = agent.env.step(action)
            done = terminated or truncated
        print(info)


def train_agent__mc(agent: BaseAgent, tensorboard, trial):
    running_reward = deque(maxlen=50)
    episode_count = 0
    for episode, info, step in iter_episodes__monte_carlo(agent):
        if "episode" in info:
            episode_count += 1
            running_reward.append(info["episode"]["r"])
            mean_reward = np.array(running_reward).mean()
            tensorboard.write_episode_info(step, info, episode_count=episode_count, mean_reward=mean_reward)
            if trial and len(running_reward) > 50:
                trial.report(np.array(running_reward).mean(), step)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            agent.update_network(episode)
            if mean_reward >= agent.env.spec.reward_threshold:
                print("Solved!!")
                agent.save_weights()
                break
    return running_reward


def train_agent__td(agent: BaseAgent, tensorboard, trial):
    running_reward = deque(maxlen=50)
    episode_count = 0
    for info, step in iter_episodes__td(agent):
        if "episode" in info:
            episode_count += 1
            running_reward.append(info["episode"]["r"])
            mean_reward = np.array(running_reward).mean()
            tensorboard.write_episode_info(step, info, episode_count=episode_count, mean_reward=mean_reward)
            if trial and len(running_reward) > 50:
                trial.report(np.array(running_reward).mean(), step)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            threshold = agent.env.spec.reward_threshold
            if mean_reward >= threshold:
                print("Solved!!")
                agent.save_weights()
                break
        agent.update_network()
    return running_reward
