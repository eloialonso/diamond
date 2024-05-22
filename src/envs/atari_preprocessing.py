"""
Derived from https://github.com/openai/gym/blob/master/gym/wrappers/atari_preprocessing.py
Implementation of Atari 2600 Preprocessing following the guidelines of Machado et al., 2018.
"""

from __future__ import annotations

from typing import Any, SupportsFloat

import cv2
import numpy as np

import gymnasium as gym
from gymnasium.core import WrapperActType, WrapperObsType
from gymnasium.spaces import Box


class AtariPreprocessing(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym.Env,
        noop_max: int,
        frame_skip: int,
        screen_size: int,
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self,
            noop_max=noop_max,
            frame_skip=frame_skip,
            screen_size=screen_size,
        )
        gym.Wrapper.__init__(self, env)

        assert frame_skip > 0
        assert screen_size > 0
        assert noop_max >= 0
        if frame_skip > 1 and getattr(env.unwrapped, "_frameskip", None) != 1:
            raise ValueError(
                "Disable frame-skipping in the original env. Otherwise, more than one frame-skip will happen as through this wrapper"
            )
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

        self.frame_skip = frame_skip
        self.screen_size = screen_size

        # buffer of most recent two observations for max pooling
        assert isinstance(env.observation_space, Box)
        self.obs_buffer = [
            np.empty(env.observation_space.shape, dtype=np.uint8),
            np.empty(env.observation_space.shape, dtype=np.uint8),
        ]

        self.lives = 0
        self.game_over = False

        _low, _high, _obs_dtype = (0, 255, np.uint8)
        _shape = (screen_size, screen_size, 3)
        self.observation_space = Box(low=_low, high=_high, shape=_shape, dtype=_obs_dtype)

    @property
    def ale(self):
        """Make ale as a class property to avoid serialization error."""
        return self.env.unwrapped.ale

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        total_reward, terminated, truncated, info = 0.0, False, False, {}

        life_loss = False

        for t in range(self.frame_skip):
            _, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            self.game_over = terminated

            if self.ale.lives() < self.lives:
                life_loss = True
                self.lives = self.ale.lives()

            if terminated or truncated:
                break

            if t == self.frame_skip - 2:
                self.ale.getScreenRGB(self.obs_buffer[1])
            elif t == self.frame_skip - 1:
                self.ale.getScreenRGB(self.obs_buffer[0])

        info["life_loss"] = life_loss

        obs, original_obs = self._get_obs()
        info["original_obs"] = original_obs

        return obs, total_reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Resets the environment using preprocessing."""
        # NoopReset
        _, reset_info = self.env.reset(seed=seed, options=options)

        reset_info["life_loss"] = False

        noops = self.env.unwrapped.np_random.integers(1, self.noop_max + 1) if self.noop_max > 0 else 0
        for _ in range(noops):
            _, _, terminated, truncated, step_info = self.env.step(0)
            reset_info.update(step_info)
            if terminated or truncated:
                _, reset_info = self.env.reset(seed=seed, options=options)

        self.lives = self.ale.lives()
        self.ale.getScreenRGB(self.obs_buffer[0])
        self.obs_buffer[1].fill(0)

        obs, original_obs = self._get_obs()
        reset_info["original_obs"] = original_obs

        return obs, reset_info

    def _get_obs(self):
        if self.frame_skip > 1:  # more efficient in-place pooling
            np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])

        original_obs = self.obs_buffer[0]
        obs = cv2.resize(
            original_obs,
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA,
        )

        obs = np.asarray(obs, dtype=np.uint8)

        return obs, original_obs
