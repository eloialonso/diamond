from collections import Counter
import multiprocessing as mp
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset

from .episode import Episode
from .segment import Segment, SegmentId
from .utils import make_segment
from utils import StateDictMixin


class Dataset(StateDictMixin, TorchDataset):
    def __init__(
        self,
        directory: Path,
        dataset_full_res: Optional[TorchDataset],
        name: Optional[str] = None,
        cache_in_ram: bool = False,
        use_manager: bool = False,
        save_on_disk: bool = True,
    ) -> None:
        super().__init__()

        # State
        self.is_static = False
        self.num_episodes = None
        self.num_steps = None
        self.start_idx = None
        self.lengths = None
        self.counter_rew = None
        self.counter_end = None

        self._directory = Path(directory).expanduser()
        self._name = name if name is not None else self._directory.stem
        self._cache_in_ram = cache_in_ram
        self._save_on_disk = save_on_disk
        self._default_path = self._directory / "info.pt"
        self._cache = mp.Manager().dict() if use_manager else {}
        self._reset()

        self._dataset_full_res = dataset_full_res

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, segment_id: SegmentId) -> Segment:
        episode = self.load_episode(segment_id.episode_id)
        segment = make_segment(episode, segment_id, should_pad=True)
        if self._dataset_full_res is not None:
            segment_id_full_res = SegmentId(episode.info["original_file_id"], segment_id.start, segment_id.stop)
            segment.info["full_res"] = self._dataset_full_res[segment_id_full_res].obs
        elif "full_res" in segment.info:
            segment.info["full_res"] = segment.info["full_res"][segment_id.start:segment_id.stop]
        return segment
        
    def __str__(self) -> str:
        return f"{self.name}: {self.num_episodes} episodes, {self.num_steps} steps."

    @property
    def name(self) -> str:
        return self._name

    @property
    def counts_rew(self) -> List[int]:
        return [self.counter_rew[r] for r in [-1, 0, 1]]

    @property
    def counts_end(self) -> List[int]:
        return [self.counter_end[e] for e in [0, 1]]

    def _reset(self) -> None:
        self.num_episodes = 0
        self.num_steps = 0
        self.start_idx = np.array([], dtype=np.int64)
        self.lengths = np.array([], dtype=np.int64)
        self.counter_rew = Counter()
        self.counter_end = Counter()
        self._cache.clear()

    def clear(self) -> None:
        self.assert_not_static()
        if self._directory.is_dir():
            shutil.rmtree(self._directory)
        self._reset()

    def load_episode(self, episode_id: int) -> Episode:
        if self._cache_in_ram and episode_id in self._cache:
            episode = self._cache[episode_id]
        else:
            episode = Episode.load(self._get_episode_path(episode_id))
            if self._cache_in_ram:
                self._cache[episode_id] = episode
        return episode

    def add_episode(self, episode: Episode, *, episode_id: Optional[int] = None) -> int:
        self.assert_not_static()
        episode = episode.to("cpu")

        if episode_id is None:
            episode_id = self.num_episodes
            self.start_idx = np.concatenate((self.start_idx, np.array([self.num_steps])))
            self.lengths = np.concatenate((self.lengths, np.array([len(episode)])))
            self.num_steps += len(episode)
            self.num_episodes += 1

        else:
            assert episode_id < self.num_episodes
            old_episode = self.load_episode(episode_id)
            incr_num_steps = len(episode) - len(old_episode)
            self.lengths[episode_id] = len(episode)
            self.start_idx[episode_id + 1 :] += incr_num_steps
            self.num_steps += incr_num_steps
            self.counter_rew.subtract(old_episode.rew.sign().tolist())
            self.counter_end.subtract(old_episode.end.tolist())

        self.counter_rew.update(episode.rew.sign().tolist())
        self.counter_end.update(episode.end.tolist())

        if self._save_on_disk:
            episode.save(self._get_episode_path(episode_id))

        if self._cache_in_ram:
            self._cache[episode_id] = episode

        return episode_id

    def _get_episode_path(self, episode_id: int) -> Path:
        n = 3  # number of hierarchies
        powers = np.arange(n)
        subfolders = np.floor((episode_id % 10 ** (1 + powers)) / 10**powers) * 10**powers
        subfolders = [int(x) for x in subfolders[::-1]]
        subfolders = "/".join([f"{x:0{n - i}d}" for i, x in enumerate(subfolders)])
        return self._directory / subfolders / f"{episode_id}.pt"

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._cache.clear()

    def assert_not_static(self) -> None:
        assert not self.is_static, "Trying to modify a static dataset."

    def save_to_default_path(self) -> None:
        self._default_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.state_dict(), self._default_path)

    def load_from_default_path(self) -> None:
        if self._default_path.is_file():
            self.load_state_dict(torch.load(self._default_path))


class CSGOHdf5Dataset(StateDictMixin, TorchDataset):
    def __init__(self, directory: Path) -> None:
        super().__init__()
        filenames = sorted(Path(directory).rglob("*.hdf5"), key=lambda x: int(x.stem.split("_")[-1]))
        self._filenames = {f"{x.parent.name}/{x.name}": x for x in filenames}
        self._length_one_episode = 1000
        self.num_episodes = len(self._filenames)
        self.num_steps = self._length_one_episode * self.num_episodes
        self.lengths = np.array([self._length_one_episode] * self.num_episodes, dtype=np.int64)

    def __len__(self) -> int:
        return self.num_steps
  
    def save_to_default_path(self) -> None:
        pass

    def __getitem__(self, segment_id: SegmentId) -> Segment:
        assert segment_id.start < self._length_one_episode and segment_id.stop > 0 and segment_id.start < segment_id.stop
        pad_len_right = max(0, segment_id.stop - self._length_one_episode)
        pad_len_left = max(0, -segment_id.start)

        start = max(0, segment_id.start)
        stop = min(self._length_one_episode, segment_id.stop)
        mask_padding = torch.cat((torch.zeros(pad_len_left), torch.ones(stop - start), torch.zeros(pad_len_right))).bool()

        with h5py.File(self._filenames[segment_id.episode_id], "r") as f:
            obs = torch.stack([torch.tensor(f[f"frame_{i}_x"][:]).flip(2).permute(2, 0, 1).div(255).mul(2).sub(1) for i in range(start, stop)])
            act = torch.tensor(np.array([f[f"frame_{i}_y"][:] for i in range(start, stop)]))

        def pad(x):
            right = F.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [pad_len_right]) if pad_len_right > 0 else x
            return F.pad(right, [0 for _ in range(2 * x.ndim - 2)] + [pad_len_left, 0]) if pad_len_left > 0 else right

        obs = pad(obs)
        act = pad(act)
        rew = torch.zeros(obs.size(0))
        end = torch.zeros(obs.size(0), dtype=torch.uint8)
        trunc = torch.zeros(obs.size(0), dtype=torch.uint8)
        return Segment(obs, act, rew, end, trunc, mask_padding, info={}, id=SegmentId(segment_id.episode_id, start, stop))
    
    def load_episode(self, episode_id: int) -> Episode:  # used by DatasetTraverser
        s = self[SegmentId(episode_id, 0, self._length_one_episode)]
        return Episode(s.obs, s.act, s.rew, s.end, s.trunc, s.info)