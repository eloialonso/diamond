from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Union

import torch


@dataclass
class SegmentId:
    episode_id: Union[int, str]
    start: int
    stop: int


@dataclass
class Segment:
    obs: torch.FloatTensor
    act: torch.LongTensor
    rew: torch.FloatTensor
    end: torch.ByteTensor
    trunc: torch.ByteTensor
    mask_padding: torch.BoolTensor
    info: Dict[str, Any]
    id: SegmentId

    @property
    def effective_size(self):
        return self.mask_padding.sum().item()
