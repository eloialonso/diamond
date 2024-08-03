from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor

from .denoiser import DDPMDenoiser


def add_dims(input: Tensor, n: int) -> Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))


@dataclass
class DDPMSamplerConfig:
    num_steps_denoising: int


class DDPMSampler:
    def __init__(self, denoiser: DDPMDenoiser, cfg: DDPMSamplerConfig):
        self.denoiser = denoiser
        self.cfg = cfg

    @torch.no_grad()
    def sample_next_obs(self, obs: Tensor, act: Tensor) -> Tuple[Tensor, List[Tensor]]:
        device = obs.device
        scheduler = self.denoiser.ddpm_scheduler
        b, t, c, h, w = obs.size()
        obs = obs.reshape(b, t * c, h, w)
        x = torch.randn(b, c, h, w, device=device)
        trajectory = [x]

        # Default DDPM with 1-step denoising would set `timestep = num_train_timesteps`.
        # We found that the network learnt identity function when timestep is set to high values,
        # since it has to predict the noise (more details in section 5.1 of our paper - https://arxiv.org/pdf/2405.12399).
        # We found experimentally that the best 1-step setup for DDPM was to use `timestep = num_train_timesteps // 2` instead.
        if self.cfg.num_steps_denoising == 1:
            half = scheduler.config.num_train_timesteps // 2
            timestep_ = torch.full(fill_value=half, size=(b,), dtype=torch.long, device=device)
            predicted_noise = self.denoiser(x, timestep_, obs, act)
            
            alphas_cumprod = scheduler.alphas_cumprod.to(device)[timestep_]
            sqrt_alpha_prod = add_dims(alphas_cumprod ** 0.5, x.ndim)
            sqrt_one_minus_alpha_prod = add_dims((1 - alphas_cumprod) ** 0.5, x.ndim)
            denoised = (x - sqrt_one_minus_alpha_prod * predicted_noise) / sqrt_alpha_prod
            # Quantize to {0, ..., 255}, then back to [-1, 1]
            denoised = denoised.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
            
            trajectory.append(denoised)
            return denoised, trajectory
        
        else:
            scheduler.set_timesteps(self.cfg.num_steps_denoising)
            for timestep in scheduler.timesteps:
                timestep_ = torch.full(fill_value=timestep, size=(b,), device=device)
                predicted_noise = self.denoiser(x, timestep_, obs, act)
                x = scheduler.step(predicted_noise, timestep, x).prev_sample
                trajectory.append(x)
            x = x.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
            return x, trajectory
