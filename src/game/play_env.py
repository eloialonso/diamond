from collections import defaultdict, namedtuple
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pygame
import torch
from torch import Tensor

from agent import Agent
from data import Dataset, Episode
from game.keymap import ActionNames, Keymap
from envs import WorldModelEnv


NamedEnv = namedtuple("NamedEnv", "name env")
OneStepData = namedtuple("OneStepData", "obs act rew end trunc")


class PlayEnv:
    def __init__(
        self,
        agent: Agent,
        envs: List[NamedEnv],
        action_names: ActionNames,
        keymap: Keymap,
        recording_mode: bool,
        store_denoising_trajectory: bool,
        store_original_obs: bool,
    ) -> None:
        self.agent = agent
        self.envs = envs
        self.action_names = action_names
        self.keymap = keymap
        self.recording_mode = recording_mode
        self.store_denoising_trajectory = store_denoising_trajectory
        self.store_original_obs = store_original_obs
        self.is_human_player = False
        self.env_id = 0
        self.env_name, self.env = self.envs[0]
        self.obs, self.t, self.return_, self.hx_cx, self.ckpt_id, self.buffer, self.rec_dataset = (None,) * 7

    def print_controls(self) -> None:
        print("\nControls (play mode):\n")
        print("m : controller (policy/human)")
        print("↑ : imagination horizon (+1)")
        print("↓ : imagination horizon (-1)")
        print(f"→ : next environment ({' → '.join([env_name for (env_name, _) in self.envs])})")
        print(f"← : prev environment ({' ← '.join([env_name for (env_name, _) in self.envs])})")
        print("\nEnvironment actions:\n")
        for key, idx in self.keymap.items():
            key_name = pygame.key.name(key)
            print(f"{'⎵' if key_name == 'space' else key_name} : {self.action_names[idx]}")

    def next_mode(self) -> bool:
        self.switch_controller()
        return True

    def next_axis_1(self) -> bool:
        self.update_wm_horizon(+1)
        return True

    def prev_axis_1(self) -> bool:
        self.update_wm_horizon(-1)
        return True

    def next_axis_2(self) -> bool:
        self.switch_env(self.env_id + 1)
        return True

    def prev_axis_2(self) -> bool:
        self.switch_env(self.env_id - 1)
        return True

    def is_wm_env(self) -> bool:
        return isinstance(self.env, WorldModelEnv)

    def print_env(self) -> None:
        print(f"> Environment: {self.env_name}")

    def print_control(self) -> None:
        print(f"> Control: {'human' if self.is_human_player else 'policy'}")

    def switch_env(self, env_id: int) -> None:
        self.env_id = env_id % len(self.envs)
        self.env_name, self.env = self.envs[self.env_id]
        self.print_env()

    def switch_controller(self) -> None:
        self.is_human_player = not self.is_human_player
        self.print_control()

    def update_wm_horizon(self, incr: int) -> None:
        if self.is_wm_env():
            self.env.horizon = max(1, self.env.horizon + incr)

    def reset_recording(self) -> None:
        self.buffer = defaultdict(list)
        self.buffer["info"] = defaultdict(list)
        dir = Path("dataset") / f"rec_{self.env_name}_{'H' if self.is_human_player else 'π'}"
        self.rec_dataset = Dataset(dir)
        self.rec_dataset.load_from_default_path()

    def reset(self) -> Tuple[Tensor, None]:
        self.obs, _ = self.env.reset()
        self.t, self.return_, self.hx_cx = 0, 0, None
        if self.recording_mode:
            self.reset_recording()
        return self.obs, None

    @torch.no_grad()
    def step(self, act: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        if self.is_human_player:
            act = torch.tensor([act], device=self.agent.device)
        else:
            logits_act, value, self.hx_cx = self.agent.actor_critic(self.obs, self.hx_cx)
            dst = torch.distributions.categorical.Categorical(logits=logits_act)
            act = dst.sample()
            entropy = dst.entropy() / math.log(2)
        entropy = None if self.is_human_player else f"{entropy.item():.2f}"
        value = None if self.is_human_player else f"{value.item():.2f}"
        next_obs, rew, end, trunc, env_info = self.env.step(act)
        data = OneStepData(self.obs, act, rew, end, trunc)
        self.return_ += rew.item()
        control = "human" if self.is_human_player else "policy"
        header = [
            [
                f"Env     : {self.env_name}",
                f"Control : {control}",
                f"Timestep: {self.t + 1}",
                f"Horizon : {self.env.horizon}" if self.is_wm_env() else "",
            ],
            [
                f"Trunc : {bool(trunc)}",
                f"Done  : {bool(end)}",
                f"Reward: {rew.item():.2f}",
                f"Return: {self.return_:.2f}",
            ],
            [
                f"Action : {self.action_names[act[0]]}",
                f"Entropy: {entropy}",
                f"Value  : {value}",
            ],
        ]
        info = {"header": header}

        if end or trunc:
            d = "Dead" if end else ("Horizon" if self.is_wm_env() else "Timed out")
            print(f"[{d}] return = {self.return_} - length = {self.t + 1}.")

        if self.recording_mode:
            for k, v in data._asdict().items():
                self.buffer[k].append(v)
            if self.store_denoising_trajectory and "denoising_trajectory" in env_info:
                self.buffer["info"]["denoising_trajectory"].append(env_info["denoising_trajectory"])
            if self.store_original_obs and "original_obs" in env_info:
                original_obs = (torch.tensor(env_info["original_obs"][0]).permute(2, 0, 1).unsqueeze(0).contiguous())
                self.buffer["info"]["original_obs"].append(original_obs)
            if end or trunc:
                ep_dict = {k: torch.cat(v, dim=0) for k, v in self.buffer.items() if k != "info"}
                ep_info = {k: torch.cat(v, dim=0) for k, v in self.buffer["info"].items()}
                ep = Episode(**ep_dict, info=ep_info).to("cpu")
                self.rec_dataset.add_episode(ep)
                self.rec_dataset.save_to_default_path()

        self.obs = next_obs
        self.t += 1

        return next_obs, rew, end, trunc, info
