from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from data import Batch
from .inner_model import InnerModel, InnerModelConfig
from utils import ComputeLossOutput


def add_dims(input: Tensor, n: int) -> Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))


@dataclass
class SigmaDistributionConfig:
    loc: float
    scale: float
    sigma_min: float
    sigma_max: float


@dataclass
class DenoiserConfig:
    inner_model: InnerModelConfig
    sigma_data: float
    sigma_offset_noise: float


class Denoiser(nn.Module):
    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.inner_model = InnerModel(cfg.inner_model)
        self.sample_sigma_training = None

    @property
    def device(self) -> torch.device:
        return self.inner_model.noise_emb.weight.device

    def setup_training(self, cfg: SigmaDistributionConfig) -> None:
        assert self.sample_sigma_training is None

        def sample_sigma(n: int, device: torch.device):
            s = torch.randn(n, device=device) * cfg.scale + cfg.loc
            return s.exp().clip(cfg.sigma_min, cfg.sigma_max)

        self.sample_sigma_training = sample_sigma

    def forward(self, noisy_next_obs: Tensor, sigma: Tensor, obs: Tensor, act: Tensor) -> Tuple[Tensor, Tensor]:
        c_in, c_out, c_skip, c_noise = self._compute_conditioners(sigma)
        rescaled_obs = obs / self.cfg.sigma_data
        rescaled_noise = noisy_next_obs * c_in
        model_output = self.inner_model(rescaled_noise, c_noise, rescaled_obs, act)
        denoised = model_output * c_out + noisy_next_obs * c_skip
        return model_output, denoised

    @torch.no_grad()
    def denoise(self, noisy_next_obs: Tensor, sigma: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        _, d = self(noisy_next_obs, sigma, obs, act)
        # Quantize to {0, ..., 255}, then back to [-1, 1]
        d = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
        return d

    def _compute_conditioners(self, sigma: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        sigma = (sigma**2 + self.cfg.sigma_offset_noise**2).sqrt()
        c_in = 1 / (sigma**2 + self.cfg.sigma_data**2).sqrt()
        c_skip = self.cfg.sigma_data**2 / (sigma**2 + self.cfg.sigma_data**2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        return *(add_dims(c, 4) for c in (c_in, c_out, c_skip)), add_dims(c_noise, 1)

    def compute_loss(self, batch: Batch) -> ComputeLossOutput:
        n = self.cfg.inner_model.num_steps_conditioning
        seq_length = batch.obs.size(1) - n

        all_obs = batch.obs.clone()
        loss = 0

        for i in range(seq_length):
            obs = all_obs[:, i : n + i]
            next_obs = all_obs[:, n + i]
            act = batch.act[:, i : n + i]
            mask = batch.mask_padding[:, n + i]

            b, t, c, h, w = obs.shape
            obs = obs.reshape(b, t * c, h, w)

            sigma = self.sample_sigma_training(b, self.device)
            _, c_out, c_skip, _ = self._compute_conditioners(sigma)

            offset_noise = self.cfg.sigma_offset_noise * torch.randn(b, c, 1, 1, device=next_obs.device)
            noisy_next_obs = next_obs + offset_noise + torch.randn_like(next_obs) * add_dims(sigma, next_obs.ndim)

            model_output, denoised = self(noisy_next_obs, sigma, obs, act)

            target = (next_obs - c_skip * noisy_next_obs) / c_out
            loss += F.mse_loss(model_output[mask], target[mask])

            all_obs[:, n + i] = denoised.detach().clamp(-1, 1)

        loss /= seq_length
        return loss, {"loss_denoising": loss.detach()}
