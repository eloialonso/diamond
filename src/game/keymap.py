from typing import Dict, List, Tuple

import gymnasium
import pygame


ActionNames = List[str]
Keymap = Dict[int, int]


def get_keymap_and_action_names(name: str) -> Tuple[Keymap, ActionNames]:
    if name == "empty":
        return EMPTY_KEYMAP, EMPTY_ACTION_NAMES

    if name == "dataset_mode":
        return DATASET_MODE_KEYMAP, DATASET_MODE_ACTION_NAMES

    if name == "atari":
        return ATARI_KEYMAP, ATARI_ACTION_NAMES

    assert name.startswith("atari/")
    env_id = name.split("atari/")[1]
    action_names = [x.lower() for x in gymnasium.make(env_id).unwrapped.get_action_meanings()]
    keymap = {}
    for key, value in ATARI_KEYMAP.items():
        if ATARI_ACTION_NAMES[value] in action_names:
            keymap[key] = action_names.index(ATARI_ACTION_NAMES[value])
    return keymap, action_names


ATARI_ACTION_NAMES = [
    "noop",
    "fire",
    "up",
    "right",
    "left",
    "down",
    "upright",
    "upleft",
    "downright",
    "downleft",
    "upfire",
    "rightfire",
    "leftfire",
    "downfire",
    "uprightfire",
    "upleftfire",
    "downrightfire",
    "downleftfire",
]

ATARI_KEYMAP = {
    pygame.K_SPACE: 1,
    pygame.K_w: 2,
    pygame.K_d: 3,
    pygame.K_a: 4,
    pygame.K_s: 5,
    pygame.K_t: 6,
    pygame.K_r: 7,
    pygame.K_g: 8,
    pygame.K_f: 9,
    pygame.K_UP: 10,
    pygame.K_RIGHT: 11,
    pygame.K_LEFT: 12,
    pygame.K_DOWN: 13,
    pygame.K_i: 14,
    pygame.K_u: 15,
    pygame.K_k: 16,
    pygame.K_j: 17,
}

DATASET_MODE_ACTION_NAMES = [
    "noop",
    "previous",
    "next",
    "previous_10",
    "next_10",
]

DATASET_MODE_KEYMAP = {
    pygame.K_LEFT: 1,
    pygame.K_RIGHT: 2,
    pygame.K_PAGEDOWN: 3,
    pygame.K_PAGEUP: 4,
}

EMPTY_ACTION_NAMES = [
    "noop",
]

EMPTY_KEYMAP = {}
