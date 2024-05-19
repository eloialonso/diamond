from collections import defaultdict
from dataclasses import dataclass
from typing import Generator, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from . import coroutine
from data import Episode, Dataset
from envs import TorchEnv
from .env_loop import make_env_loop
from utils import Logs


@coroutine
def make_collector(
    env: TorchEnv,
    model: nn.Module,
    dataset: Dataset,
    epsilon: float = 0.0,
    reset_every_collect: bool = False,
    verbose: bool = True,
) -> Generator[Logs, int, None]:
    num_envs = env.num_envs

    env_loop, buffer, episode_ids, dead = (None,) * 4
    num_steps, num_episodes, to_log, pbar = (None,) * 4

    def setup_new_collect():
        nonlocal num_steps, num_episodes, buffer, to_log, pbar
        num_steps = 0
        num_episodes = 0
        buffer = defaultdict(list)
        to_log = []
        pbar = tqdm(
            total=num_to_collect.total,
            unit=num_to_collect.unit,
            desc=f"Collect {dataset.name}",
            disable=not verbose,
        )

    def reset():
        nonlocal env_loop, episode_ids, dead
        env_loop = make_env_loop(env, model, epsilon)
        episode_ids = defaultdict(lambda: None)
        dead = [None] * num_envs

    num_to_collect = yield
    setup_new_collect()
    reset()

    while True:
        with torch.no_grad():
            all_obs, act, rew, end, trunc, *_, [infos] = env_loop.send(1)

        num_steps += num_envs
        pbar.update(num_envs if num_to_collect.steps is not None else 0)

        for i, (o, a, r, e, t) in enumerate(zip(all_obs, act, rew, end, trunc)):
            buffer[i].append((o, a, r, e, t))
            dead[i] = (e + t).clip(max=1).item()

        num_episodes += sum(dead)

        can_stop = num_to_collect.can_stop(num_steps, num_episodes)

        count_dead = 0
        for i in range(num_envs):
            # Store incomplete episodes only when reset_every_collect is set to False (train)
            add_to_dataset = dead[i] or (can_stop and not reset_every_collect)
            if add_to_dataset:
                info = {"final_observation": infos["final_observation"][count_dead]} if dead[i] else {}
                ep = Episode(*(torch.cat(x, dim=0) for x in zip(*buffer[i])), info).to("cpu")
                if episode_ids[i] is not None:
                    ep = dataset.load_episode(episode_ids[i]) + ep
                episode_ids[i] = dataset.add_episode(ep, episode_id=episode_ids[i])

            if dead[i]:
                to_log.append(
                    {
                        f"{dataset.name}/episode_id": episode_ids[i],
                        **ep.compute_metrics(),
                    }
                )
                buffer[i] = []
                episode_ids[i] = None
                pbar.update(1 if num_to_collect.episodes is not None else 0)

            count_dead += dead[i]

        if can_stop:
            pbar.close()
            metrics = {
                "num_steps": dataset.num_steps,
                "counts/rew_-1": dataset.counts_rew[0],
                "counts/rew__0": dataset.counts_rew[1],
                "counts/rew_+1": dataset.counts_rew[2],
                "counts/end_0": dataset.counts_end[0],
                "counts/end_1": dataset.counts_end[1],
            }
            to_log.append({f"{dataset.name}/{k}": v for k, v in metrics.items()})
            num_to_collect = yield to_log
            setup_new_collect()
            if reset_every_collect:
                reset()


@dataclass
class NumToCollect:
    steps: Optional[int] = None
    episodes: Optional[int] = None

    def __post_init__(self) -> None:
        assert (self.steps is None) != (self.episodes is None)

    def can_stop(self, num_steps: int, num_episodes: int) -> bool:
        return num_steps >= self.steps if self.steps is not None else num_episodes >= self.episodes

    @property
    def unit(self) -> str:
        return "steps" if self.steps is not None else "eps"

    @property
    def total(self) -> int:
        return self.steps if self.steps is not None else self.episodes
