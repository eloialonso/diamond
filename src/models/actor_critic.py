from collections import namedtuple
from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

from .blocks import Conv3x3, SmallResBlock
from coroutines.env_loop import make_env_loop
from envs import TorchEnv, WorldModelEnv
from utils import init_lstm, LossAndLogs


ActorCriticOutput = namedtuple("ActorCriticOutput", "logits_act val hx_cx")


@dataclass
class ActorCriticLossConfig:
    backup_every: int
    gamma: float
    lambda_: float
    weight_value_loss: float
    weight_entropy_loss: float


@dataclass
class ActorCriticConfig:
    lstm_dim: int
    img_channels: int
    img_size: int
    channels: List[int]
    down: List[int]
    num_actions: Optional[int] = None


class ActorCritic(nn.Module):
    def __init__(self, cfg: ActorCriticConfig) -> None:
        super().__init__()
        self.encoder = ActorCriticEncoder(cfg)
        self.lstm_dim = cfg.lstm_dim
        input_dim_lstm = cfg.channels[-1] * (cfg.img_size // 2 ** (sum(cfg.down))) ** 2
        self.lstm = nn.LSTMCell(input_dim_lstm, cfg.lstm_dim)
        self.critic_linear = nn.Linear(cfg.lstm_dim, 1)
        self.actor_linear = nn.Linear(cfg.lstm_dim, cfg.num_actions)

        self.actor_linear.weight.data.fill_(0)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data.fill_(0)
        self.critic_linear.bias.data.fill_(0)
        init_lstm(self.lstm)

        self.env_loop = None
        self.loss_cfg = None

    @property
    def device(self) -> torch.device:
        return self.lstm.weight_hh.device

    def setup_training(self, rl_env: Union[TorchEnv, WorldModelEnv], loss_cfg: ActorCriticLossConfig) -> None:
        assert self.env_loop is None and self.loss_cfg is None
        self.env_loop = make_env_loop(rl_env, self)
        self.loss_cfg = loss_cfg

    def predict_act_value(self, obs: Tensor, hx_cx: Tuple[Tensor, Tensor]) -> ActorCriticOutput:
        assert obs.ndim == 4
        x = self.encoder(obs)
        x = x.flatten(start_dim=1)
        hx, cx = self.lstm(x, hx_cx)
        return ActorCriticOutput(self.actor_linear(hx), self.critic_linear(hx).squeeze(dim=1), (hx, cx))

    def forward(self) -> LossAndLogs:
        c = self.loss_cfg
        _, act, rew, end, trunc, logits_act, val, val_bootstrap, _ = self.env_loop.send(c.backup_every)

        d = Categorical(logits=logits_act)
        entropy = d.entropy().mean()

        lambda_returns = compute_lambda_returns(rew, end, trunc, val_bootstrap, c.gamma, c.lambda_)

        loss_actions = (-d.log_prob(act) * (lambda_returns - val).detach()).mean()
        loss_values = c.weight_value_loss * F.mse_loss(val, lambda_returns)
        loss_entropy = -c.weight_entropy_loss * entropy

        loss = loss_actions + loss_entropy + loss_values

        metrics = {
            "policy_entropy": entropy.detach() / math.log(2),
            "loss_actions": loss_actions.detach(),
            "loss_entropy": loss_entropy.detach(),
            "loss_values": loss_values.detach(),
            "loss_total": loss.detach(),
        }

        return loss, metrics


class ActorCriticEncoder(nn.Module):
    def __init__(self, cfg: ActorCriticConfig) -> None:
        super().__init__()
        assert len(cfg.channels) == len(cfg.down)
        encoder_layers = [Conv3x3(cfg.img_channels, cfg.channels[0])]
        for i in range(len(cfg.channels)):
            encoder_layers.append(SmallResBlock(cfg.channels[max(0, i - 1)], cfg.channels[i]))
            if cfg.down[i]:
                encoder_layers.append(nn.MaxPool2d(2))
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


@torch.no_grad()
def compute_lambda_returns(
    rew: Tensor,
    end: Tensor,
    trunc: Tensor,
    val_bootstrap: Tensor,
    gamma: float,
    lambda_: float,
) -> Tensor:
    assert rew.ndim == 2 and rew.size() == end.size() == trunc.size() == val_bootstrap.size()

    rew = rew.sign()  # clip reward

    end_or_trunc = (end + trunc).clip(max=1)
    not_end = 1 - end
    not_trunc = 1 - trunc

    lambda_returns = rew + not_end * gamma * (not_trunc * (1 - lambda_) + trunc) * val_bootstrap

    if lambda_ == 0:
        return lambda_returns

    last = val_bootstrap[:, -1]
    for t in reversed(range(rew.size(1))):
        lambda_returns[:, t] += end_or_trunc[:, t].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, t]

    return lambda_returns
