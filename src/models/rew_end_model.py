from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_confusion_matrix

from .blocks import Conv3x3, Downsample, ResBlocks
from data import Batch
from utils import init_lstm, LossAndLogs


@dataclass
class RewEndModelConfig:
    lstm_dim: int
    img_channels: int
    img_size: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[int]
    num_actions: Optional[int] = None


class RewEndModel(nn.Module):
    def __init__(self, cfg: RewEndModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = RewEndEncoder(2 * cfg.img_channels, cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)
        self.act_emb = nn.Embedding(cfg.num_actions, cfg.cond_channels)
        input_dim_lstm = cfg.channels[-1] * (cfg.img_size // 2 ** (len(cfg.depths) - 1)) ** 2
        self.lstm = nn.LSTM(input_dim_lstm, cfg.lstm_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(cfg.lstm_dim, cfg.lstm_dim),
            nn.SiLU(),
            nn.Linear(cfg.lstm_dim, 3 + 2, bias=False),
        )
        init_lstm(self.lstm)

    def predict_rew_end(
        self,
        obs: Tensor,
        act: Tensor,
        next_obs: Tensor,
        hx_cx: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
        b, t, c, h, w = obs.shape
        obs, act, next_obs = obs.reshape(b * t, c, h, w), act.reshape(b * t), next_obs.reshape(b * t, c, h, w)
        x = self.encoder(torch.cat((obs, next_obs), dim=1), self.act_emb(act))
        x = x.reshape(b, t, -1)  # (b t) e h w -> b t (e h w)
        x, hx_cx = self.lstm(x, hx_cx)
        logits = self.head(x)
        return logits[:, :, :-2], logits[:, :, -2:], hx_cx

    def forward(self, batch: Batch) -> LossAndLogs:
        obs = batch.obs[:, :-1]
        act = batch.act[:, :-1]
        next_obs = batch.obs[:, 1:]
        rew = batch.rew[:, :-1]
        end = batch.end[:, :-1]
        mask = batch.mask_padding[:, :-1]

        # When dead, replace frame (gray padding) by true final obs
        dead = end.bool().any(dim=1)
        if dead.any():
            final_obs = torch.stack([i["final_observation"] for i, d in zip(batch.info, dead) if d]).to(obs.device)
            next_obs[dead, end[dead].argmax(dim=1)] = final_obs

        logits_rew, logits_end, _ = self.predict_rew_end(obs, act, next_obs)
        logits_rew = logits_rew[mask]
        logits_end = logits_end[mask]
        target_rew = rew[mask].sign().long().add(1)  # clipped to {-1, 0, 1}
        target_end = end[mask]

        loss_rew = F.cross_entropy(logits_rew, target_rew)
        loss_end = F.cross_entropy(logits_end, target_end)
        loss = loss_rew + loss_end

        metrics = {
            "loss_rew": loss_rew.detach(),
            "loss_end": loss_end.detach(),
            "loss_total": loss.detach(),
            "confusion_matrix": {
                "rew": multiclass_confusion_matrix(logits_rew, target_rew, num_classes=3),
                "end": multiclass_confusion_matrix(logits_end, target_end, num_classes=2),
            },
        }
        return loss, metrics


class RewEndEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        depths: List[int],
        channels: List[int],
        attn_depths: List[int],
    ) -> None:
        super().__init__()
        assert len(depths) == len(channels) == len(attn_depths)
        self.conv_in = Conv3x3(in_channels, channels[0])
        blocks = []
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            blocks.append(
                ResBlocks(
                    list_in_channels=[c1] + [c2] * (n - 1),
                    list_out_channels=[c2] * n,
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                )
            )
        blocks.append(
            ResBlocks(
                list_in_channels=[channels[-1]] * 2,
                list_out_channels=[channels[-1]] * 2,
                cond_channels=cond_channels,
                attn=True,
            )
        )
        self.blocks = nn.ModuleList(blocks)
        self.downsamples = nn.ModuleList([nn.Identity()] + [Downsample(c) for c in channels[:-1]] + [nn.Identity()])

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x = self.conv_in(x)
        for block, down in zip(self.blocks, self.downsamples):
            x = down(x)
            x, _ = block(x, cond)
        return x
