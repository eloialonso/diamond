from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor

from data import Dataset


class DatasetEnv:
    def __init__(self, datasets: List[Dataset], action_names: List[str]) -> None:
        self.datasets = [d for d in datasets if len(d) > 0]
        assert len(self.datasets) > 0
        self.action_names = action_names
        self.dataset_id = 0
        self.dataset = self.datasets[0]
        self.episode_id = None
        self.episode = None
        self.t = None
        self.ep_return = None
        self.ep_length = None
        self.pos_return = None
        self.neg_return = None
        self.load_episode(0)

    def print_controls(self) -> None:
        print("\nControls (dataset mode):\n")
        print(f"m : datasets ({'/'.join([d.name for d in self.datasets])})")
        print("↑ : next episode")
        print("↓ : prev episode")
        print("→ : next timestep")
        print("← : prev timestep")

    def next_mode(self) -> bool:
        self.switch_dataset()
        return True

    def next_axis_1(self) -> bool:
        self.load_episode(self.episode_id + 1)
        return True

    def prev_axis_1(self) -> bool:
        self.load_episode(self.episode_id - 1)
        return True

    def next_axis_2(self) -> bool:
        return False

    def prev_axis_2(self) -> bool:
        return False

    def load_episode(self, episode_id: int) -> None:
        self.episode_id = episode_id % self.dataset.num_episodes
        self.episode = self.dataset.load_episode(self.episode_id)
        self.set_timestep(0)
        metrics = self.episode.compute_metrics()
        self.ep_return = metrics["return"]
        self.ep_length = metrics["length"]
        self.pos_return = self.episode.rew[self.episode.rew > 0].sum().item()
        self.neg_return = self.episode.rew[self.episode.rew < 0].sum().abs().item()

    def set_timestep(self, timestep: int) -> None:
        self.t = timestep % len(self.episode)
        self.obs = self.episode.obs[self.t].unsqueeze(0)
        self.act = self.episode.act[self.t]
        self.rew = self.episode.rew[self.t]
        self.end = self.episode.end[self.t]
        self.trunc = self.episode.trunc[self.t]

    def switch_dataset(self) -> None:
        self.dataset_id = (self.dataset_id + 1) % len(self.datasets)
        self.dataset = self.datasets[self.dataset_id]
        self.load_episode(0)

    def reset(self) -> None:
        self.set_timestep(0)
        return self.obs, None

    @torch.no_grad()
    def step(self, act: int) -> Tuple[Tensor, Tensor, bool, bool, Dict[str, Any]]:
        match act:
            case 1:
                self.set_timestep(self.t - 1)
            case 2:
                self.set_timestep(self.t + 1)
            case 3:
                self.set_timestep(self.t - 10)
            case 4:
                self.set_timestep(self.t + 10)

        n_digits = len(str(self.ep_length))

        header = [
            [
                f"Dataset: {self.dataset.name}",
                f"Episode: {self.episode_id}",
                "--------",
                f"Return (+): +{self.pos_return:4.1f}",
                f"Return (-): -{self.neg_return:4.1f}",
                f"Total     :  {self.ep_return:4.1f}",
            ],
            [
                f"Action: {self.action_names[self.act]}",
                f"Trunc : {bool(self.trunc)}",
                f"Done  : {bool(self.end)}",
                f"Reward: {self.rew.item():.2f}",
                "-------",
                f"To here: {self.episode.rew[:self.t + 1].sum().item():.2f}",
                f"To go  : {self.episode.rew[self.t + 1:].sum().item():.2f}",
            ],
            [
                f"Timestep: {self.t:{n_digits}d}",
                f"Length  : {self.ep_length}",
            ],
        ]
        info = {"header": header}
        return self.obs, torch.tensor(0), False, False, info
