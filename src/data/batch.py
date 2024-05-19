from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from .segment import SegmentId


@dataclass
class Batch:
    obs: torch.ByteTensor
    act: torch.LongTensor
    rew: torch.FloatTensor
    end: torch.LongTensor
    trunc: torch.LongTensor
    mask_padding: torch.BoolTensor
    info: List[Dict[str, Any]]
    segment_ids: List[SegmentId]

    def pin_memory(self) -> Batch:
        return Batch(**{k: v if k in ("segment_ids", "info") else v.pin_memory() for k, v in self.__dict__.items()})

    def to(self, device: torch.device) -> Batch:
        return Batch(**{k: v if k in ("segment_ids", "info") else v.to(device) for k, v in self.__dict__.items()})
