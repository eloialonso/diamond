from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from envs import TorchEnv, WorldModelEnv
from models.actor_critic import ActorCritic, ActorCriticConfig, ActorCriticLossConfig
from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.rew_end_model import RewEndModel, RewEndModelConfig
from utils import extract_state_dict


@dataclass
class AgentConfig:
    denoiser: DenoiserConfig
    upsampler: Optional[DenoiserConfig]
    rew_end_model: Optional[RewEndModelConfig]
    actor_critic: Optional[ActorCriticConfig]
    num_actions: int

    def __post_init__(self) -> None:
        self.denoiser.inner_model.num_actions = self.num_actions
        if self.upsampler is not None:
            self.upsampler.inner_model.num_actions = self.num_actions
        if self.rew_end_model is not None:
            self.rew_end_model.num_actions = self.num_actions
        if self.actor_critic is not None:
            self.actor_critic.num_actions = self.num_actions


class Agent(nn.Module):
    def __init__(self, cfg: AgentConfig) -> None:
        super().__init__()
        self.denoiser = Denoiser(cfg.denoiser)
        self.upsampler = Denoiser(cfg.upsampler) if cfg.upsampler is not None else None
        self.rew_end_model = RewEndModel(cfg.rew_end_model) if cfg.rew_end_model is not None else None
        self.actor_critic = ActorCritic(cfg.actor_critic) if cfg.actor_critic is not None else None

    @property
    def device(self):
        return self.denoiser.device

    def setup_training(
        self,
        sigma_distribution_cfg: SigmaDistributionConfig,
        sigma_distribution_cfg_upsampler: Optional[SigmaDistributionConfig],
        actor_critic_loss_cfg: Optional[ActorCriticLossConfig],
        rl_env: Optional[Union[TorchEnv, WorldModelEnv]],
    ) -> None:
        self.denoiser.setup_training(sigma_distribution_cfg)
        if self.upsampler is not None:
            self.upsampler.setup_training(sigma_distribution_cfg_upsampler)
        if self.actor_critic is not None:
            self.actor_critic.setup_training(rl_env, actor_critic_loss_cfg)

    def load(
        self,
        path_to_ckpt: Path,
        load_denoiser: bool = True,
        load_upsampler: bool = True,
        load_rew_end_model: bool = True,
        load_actor_critic: bool = True,
    ) -> None:
        sd = torch.load(Path(path_to_ckpt), map_location=self.device)
        if load_denoiser:
            self.denoiser.load_state_dict(extract_state_dict(sd, "denoiser"))
        if load_upsampler:
            self.upsampler.load_state_dict(extract_state_dict(sd, "upsampler"))
        if load_rew_end_model and self.rew_end_model is not None:
            self.rew_end_model.load_state_dict(extract_state_dict(sd, "rew_end_model"))
        if load_actor_critic and self.actor_critic is not None:
            self.actor_critic.load_state_dict(extract_state_dict(sd, "actor_critic"))
