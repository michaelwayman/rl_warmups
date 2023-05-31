import gymnasium as gym


class BaseAgent:
    def __init__(self, params, env: gym.Env):
        self.params = params
        self.env = env

    def load_weights(self):
        raise NotImplemented

    def save_weights(self):
        raise NotImplemented

    def get_action(self, obs):
        raise NotImplemented
