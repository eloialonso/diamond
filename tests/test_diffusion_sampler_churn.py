"""Regression tests for the noise level supplied after stochastic churn."""
import math

import pytest
import torch

from models.diffusion import Denoiser, DenoiserConfig, InnerModelConfig
from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig


class GaussianDenoiser:
    """Exact posterior mean for a unit-variance Gaussian data distribution."""
    device = torch.device("cpu")

    def __init__(self):
        self.calls = []

    def denoise(self, x, sigma, prev_obs, prev_act):
        self.calls.append((x.clone(), sigma.clone()))
        return x / (1 + sigma.reshape(-1, 1, 1, 1)**2)


def conditioning(batch=2):
    return torch.zeros(batch, 1, 1, 4, 4), torch.zeros(batch, 1, dtype=torch.long)


def reference_trajectory(sampler, prev_obs):
    """EDM Euler/Heun steps, preserving DIAMOND's unit initial noise."""
    cfg = sampler.cfg
    b, _, c, h, w = prev_obs.shape
    x = torch.randn(b, c, h, w)
    trajectory = [x]
    for sigma, next_sigma in zip(sampler.sigmas[:-1], sampler.sigmas[1:]):
        gamma = min(cfg.s_churn / cfg.num_steps_denoising, math.sqrt(2) - 1)
        gamma = gamma if cfg.s_tmin <= sigma <= cfg.s_tmax else 0
        sigma_hat = sigma * (1 + gamma)
        if gamma:
            x = x + torch.randn_like(x) * cfg.s_noise * (sigma_hat**2 - sigma**2).sqrt()
        posterior = x / (1 + sigma_hat**2)
        derivative = (x - posterior) / sigma_hat
        dt = next_sigma - sigma_hat
        predicted = x + derivative * dt
        if cfg.order == 2 and next_sigma != 0:
            next_posterior = predicted / (1 + next_sigma**2)
            next_derivative = (predicted - next_posterior) / next_sigma
            x = x + (derivative + next_derivative) * (dt / 2)
        else:
            x = predicted
        trajectory.append(x)
    return trajectory


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("s_noise", [0., 1.05])
def test_churn_trajectory_matches_noise_conditioned_reference(order, s_noise):
    cfg = DiffusionSamplerConfig(4, sigma_min=.1, sigma_max=2., rho=2,
                                 order=order, s_churn=.8, s_noise=s_noise)
    sampler = DiffusionSampler(GaussianDenoiser(), cfg)
    obs, act = conditioning()
    torch.manual_seed(417)
    _, actual = sampler.sample(obs, act)
    torch.manual_seed(417)
    expected = reference_trajectory(sampler, obs)
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target)


@pytest.mark.parametrize("order", [1, 2])
def test_zero_churn_keeps_original_noise_levels(order):
    denoiser = GaussianDenoiser()
    cfg = DiffusionSamplerConfig(3, sigma_min=.1, sigma_max=2., rho=2, order=order)
    sampler = DiffusionSampler(denoiser, cfg)
    obs, act = conditioning()
    torch.manual_seed(72)
    _, trajectory = sampler.sample(obs, act)
    torch.manual_seed(72)
    expected = reference_trajectory(sampler, obs)
    for result, target in zip(trajectory, expected):
        torch.testing.assert_close(result, target)


@pytest.mark.parametrize("order", [1, 2])
def test_churn_outside_active_noise_range_is_unchanged(order):
    cfg = DiffusionSamplerConfig(3, sigma_min=.1, sigma_max=2., rho=2,
                                 order=order, s_churn=1., s_tmin=3.)
    sampler = DiffusionSampler(GaussianDenoiser(), cfg)
    obs, act = conditioning()
    torch.manual_seed(8)
    _, actual = sampler.sample(obs, act)
    torch.manual_seed(8)
    expected = reference_trajectory(sampler, obs)
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target)


def test_positive_churn_passes_increased_sigma_to_denoiser():
    denoiser = GaussianDenoiser()
    cfg = DiffusionSamplerConfig(1, sigma_max=2., s_churn=1.)
    sampler = DiffusionSampler(denoiser, cfg)
    sampler.sample(*conditioning())
    expected = sampler.sigmas[0] * math.sqrt(2)
    torch.testing.assert_close(denoiser.calls[0][1], expected)


def test_single_euler_step_matches_gaussian_posterior_at_increased_noise():
    denoiser = GaussianDenoiser()
    cfg = DiffusionSamplerConfig(1, sigma_max=2., s_churn=.4)
    sampler = DiffusionSampler(denoiser, cfg)
    result, _ = sampler.sample(*conditioning())
    noisy_input = denoiser.calls[0][0]
    sigma_hat = sampler.sigmas[0] * 1.4
    torch.testing.assert_close(result, noisy_input / (1 + sigma_hat**2))


class RecordingDenoiser(Denoiser):
    def denoise(self, x, sigma, obs, act):
        self.last_input = x.clone()
        return super().denoise(x, sigma, obs, act)


def test_single_step_with_real_denoiser_uses_matching_conditioners():
    inner = InnerModelConfig(img_channels=1, num_steps_conditioning=1, cond_channels=4,
                             depths=[1], channels=[4], attn_depths=[False], num_actions=2)
    torch.manual_seed(27)
    denoiser = RecordingDenoiser(DenoiserConfig(inner, sigma_data=.5, sigma_offset_noise=0.)).eval()
    cfg = DiffusionSamplerConfig(1, sigma_max=2., s_churn=.4)
    sampler = DiffusionSampler(denoiser, cfg)
    obs, act = conditioning()
    result, trajectory = sampler.sample(obs, act)
    noisy_input = denoiser.last_input
    expected = denoiser.denoise(noisy_input, sampler.sigmas[0] * 1.4, obs.reshape(2, 1, 4, 4), act)
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
    assert all(torch.isfinite(state).all() for state in trajectory)
