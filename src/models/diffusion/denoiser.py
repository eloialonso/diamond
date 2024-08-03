from dataclasses import dataclass
from typing import Tuple

from diffusers import DDPMScheduler
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from data import Batch
from .inner_model import InnerModel, InnerModelConfig
from utils import ComputeLossOutput


@dataclass
class DDPMDenoiserConfig:
    inner_model: InnerModelConfig
    sigma_offset_noise: float
    num_train_timesteps: int


class DDPMDenoiser(nn.Module):
    def __init__(self, cfg: DDPMDenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.noise_model = InnerModel(cfg.inner_model)
        self.ddpm_scheduler = DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps)

    @property
    def device(self) -> torch.device:
        return self.noise_model.noise_emb.weight.device

    def forward(self, noisy_next_obs: Tensor, timesteps: Tensor, obs: Tensor, act: Tensor) -> Tuple[Tensor, Tensor]:
        c_noise = (timesteps + 1) / self.cfg.num_train_timesteps
        predicted_noise = self.noise_model(noisy_next_obs, c_noise, obs, act)
        return predicted_noise

    def compute_loss(self, batch: Batch) -> ComputeLossOutput:
        assert batch.obs.size(1) == self.cfg.inner_model.num_steps_conditioning + 1
        obs = batch.obs[:, :-1]
        next_obs = batch.obs[:, -1]
        act = batch.act[:, :-1]
        mask = batch.mask_padding[:, -1]

        b, t, c, h, w = obs.shape
        obs = obs.reshape(b, t * c, h, w)

        offset_noise = self.cfg.sigma_offset_noise * torch.randn(b, c, 1, 1, device=self.device)
        noise = offset_noise + torch.randn_like(next_obs)
        timesteps = torch.randint(0, self.cfg.num_train_timesteps, size=(b,), device=self.device)
        noisy_next_obs = self.ddpm_scheduler.add_noise(next_obs, noise, timesteps)
    
        predicted_noise = self(noisy_next_obs, timesteps, obs, act)

        loss = F.mse_loss(predicted_noise[mask], noise[mask])

        return loss, {"loss_denoising": loss.detach().clone()}
