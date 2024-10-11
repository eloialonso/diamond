from collections import defaultdict, namedtuple
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pygame
import torch
from torch import Tensor

from agent import Agent
from csgo.action_processing import CSGOAction, decode_csgo_action, encode_csgo_action, print_csgo_action
from csgo.keymap import CSGO_KEYMAP
from data import Dataset, Episode
from envs import WorldModelEnv


NamedEnv = namedtuple("NamedEnv", "name env")
OneStepData = namedtuple("OneStepData", "obs act rew end trunc")


class PlayEnv:
    def __init__(
        self,
        agent: Agent,
        wm_env: WorldModelEnv,
        recording_mode: bool,
        store_denoising_trajectory: bool,
        store_original_obs: bool,
    ) -> None:
        self.agent = agent
        self.keymap = CSGO_KEYMAP
        self.recording_mode = recording_mode
        self.store_denoising_trajectory = store_denoising_trajectory
        self.store_original_obs = store_original_obs
        self.is_human_player = True
        self.env_id = 0
        self.env_name = "world model"
        self.env = wm_env
        self.obs, self.t, self.buffer, self.rec_dataset = (None,) * 4

    def print_controls(self) -> None:
        print(" m  : switch control (human/replay)")
        print("\nEnvironment actions:\n")
        for key, action_name in self.keymap.items():
            if key is not None:
                key_name = pygame.key.name(key)
                key_name = "⎵" if key_name == "space" else key_name
                print(f"{key_name} : {action_name}")

    def next_mode(self) -> bool:
        self.switch_controller()
        return True

    def next_axis_1(self) -> bool:
        return False

    def prev_axis_1(self) -> bool:
        return False

    def next_axis_2(self) -> bool:
        return False

    def prev_axis_2(self) -> bool:
        return False

    def print_env(self) -> None:
        print(f"> Environment: {self.env_name}")

    def str_control(self) -> str:
        return "human" if self.is_human_player else "replay actions (test dataset)"

    def print_control(self) -> None:
        print(f"> Control: {self.str_control()}")

    def switch_controller(self) -> None:
        self.is_human_player = not self.is_human_player
        self.print_control()

    def update_wm_horizon(self, incr: int) -> None:
        self.env.horizon = max(1, self.env.horizon + incr)

    def reset_recording(self) -> None:
        self.buffer = defaultdict(list)
        self.buffer["info"] = defaultdict(list)
        dir = Path("dataset") / f"rec_{self.env_name}_{'H' if self.is_human_player else 'R'}"
        self.rec_dataset = Dataset(dir, None)
        self.rec_dataset.load_from_default_path()

    def reset(self) -> Tuple[Tensor, None]:
        self.obs, _ = self.env.reset()
        self.t = 0
        if self.recording_mode:
            self.reset_recording()
        return self.obs, None

    @torch.no_grad()
    def step(self, csgo_action: CSGOAction) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        if self.is_human_player:
            action = encode_csgo_action(csgo_action, device=self.agent.device)
        else:
            action = self.env.next_act[self.t - 1] if self.t > 0 else self.env.act_buffer[0, -1].clone()
            csgo_action = decode_csgo_action(action.cpu())
        next_obs, rew, end, trunc, env_info = self.env.step(action)

        if not self.is_human_player and self.t == self.env.next_act.size(0):
            trunc[0] = 1

        data = OneStepData(self.obs, action, rew, end, trunc)
        keys, mouse, clicks = print_csgo_action(csgo_action)
        horizon = self.env.horizon if self.is_human_player else min(self.env.horizon, self.env.next_act.size(0))
        header = [
            [
                f"Env     : {self.env_name}",
                f"Control : {self.str_control()}",
                f"Timestep: {self.t + 1}",
                f"Horizon : {horizon}",
                "",
                f"Keys  : {keys}",
                f"Mouse : {mouse}",
                f"Clicks: {clicks}",
            ],
        ]
        info = {"header": header}
        if "obs_low_res" in env_info:
            info["obs_low_res"] = env_info["obs_low_res"]

        if self.recording_mode:
            for k, v in data._asdict().items():
                self.buffer[k].append(v)
            if "obs_low_res" in env_info:
                self.buffer["info"]["obs_low_res"].append(env_info["obs_low_res"])
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
