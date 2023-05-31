from typing import Dict

from rich import print
from torch.utils.tensorboard import SummaryWriter

from rl_warmups.common.params import BaseParams


class Tensorboard:
    def __init__(self, params: BaseParams):
        self.params = params
        self.writer = SummaryWriter(f"runs/{params.get_save_weight_path()}")

    def record_params(self):
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.params).items()])),
        )

    def write_episode_info(self, step: int, info: Dict, **kwargs):
        data = kwargs.copy()
        data["global_step"] = step
        data["episode_steps"] = info["episode"]["l"].item()
        data["episode_return"] = info["episode"]["r"].item()
        print(data)
        self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], step)
        self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], step)
