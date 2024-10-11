from typing import Generator, List, Optional

import numpy as np
import torch

from .dataset import CSGOHdf5Dataset, Dataset
from .segment import SegmentId


class BatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset: Dataset,
        rank: int,
        world_size: int,
        batch_size: int,
        seq_length: int,
        sample_weights: Optional[List[float]] = None,
        can_sample_beyond_end: bool = False,
    ) -> None:
        super().__init__(dataset)
        assert isinstance(dataset, (Dataset, CSGOHdf5Dataset))
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.sample_weights = sample_weights
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.can_sample_beyond_end = can_sample_beyond_end

    def __len__(self):
        raise NotImplementedError

    def __iter__(self) -> Generator[List[SegmentId], None, None]:
        while True:
            yield self.sample()

    def sample(self) -> List[SegmentId]:
        num_episodes = self.dataset.num_episodes

        if (self.sample_weights is None) or num_episodes < len(self.sample_weights):
            weights = self.dataset.lengths / self.dataset.num_steps
        else:
            weights = self.sample_weights
            num_weights = len(self.sample_weights)
            assert all([0 <= x <= 1 for x in weights]) and sum(weights) == 1
            sizes = [
                num_episodes // num_weights + (num_episodes % num_weights) * (i == num_weights - 1)
                for i in range(num_weights)
            ]
            weights = [w / s for (w, s) in zip(weights, sizes) for _ in range(s)]

        episodes_partition = np.arange(self.rank, num_episodes, self.world_size)
        weights = np.array(weights[self.rank::self.world_size])
        max_eps = self.batch_size
        episode_ids = np.random.choice(episodes_partition, size=max_eps, replace=True, p=weights / weights.sum())
        episode_ids = episode_ids.repeat(self.batch_size // max_eps)
        timesteps = np.random.randint(low=0, high=self.dataset.lengths[episode_ids])

        # padding allowed, both before start and after end
        if self.can_sample_beyond_end:
            starts = timesteps - np.random.randint(0, self.seq_length, len(timesteps))
            stops = starts + self.seq_length

        # padding allowed only before start
        else:
            stops = np.minimum(
                self.dataset.lengths[episode_ids], timesteps + 1 + np.random.randint(0, self.seq_length, len(timesteps))
            )
            starts = stops - self.seq_length

        return [SegmentId(*x) for x in zip(episode_ids, starts, stops)]
