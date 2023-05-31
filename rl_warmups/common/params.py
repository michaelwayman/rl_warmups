from dataclasses import dataclass


@dataclass
class BaseParams:
    seed: int = 1
    env_id: str = "LunarLander-v2"
    render: bool = False
    algo: str = "reinforce"

    n_steps: int = 500000
    learning_rate: float = 0.01
    gamma: float = 0.99

    def get_save_weight_path(self):
        return f"{self.env_id}__{self.seed}.pt"
