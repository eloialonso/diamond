"""
Credits: some parts are taken and modified from the file `config.py` from https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning/
"""

from dataclasses import dataclass 
from typing import Dict, List, Set, Tuple

import numpy as np
import pygame
import torch

from .keymap import CSGO_FORBIDDEN_COMBINATIONS, CSGO_KEYMAP


@dataclass
class CSGOAction:
    keys: List[int]
    mouse_x: float
    mouse_y: float
    l_click: bool
    r_click: bool

    def __post_init__(self) -> None:
        self.keys = filter_keys_pressed_forbidden(self.keys)
        self.process_mouse()

    @property
    def key_names(self) -> List[str]:
        return [pygame.key.name(key) for key in self.keys] 

    def process_mouse(self) -> None:
        # Clip and match mouse to closest in list of possibles
        x = np.clip(self.mouse_x, MOUSE_X_LIM[0], MOUSE_X_LIM[1])
        y = np.clip(self.mouse_y, MOUSE_Y_LIM[0], MOUSE_Y_LIM[1])
        self.mouse_x = min(MOUSE_X_POSSIBLES, key=lambda x_: abs(x_ - x))
        self.mouse_y = min(MOUSE_Y_POSSIBLES, key=lambda x_: abs(x_ - y))
        
        # Use arrows to override mouse movements
        for key in self.key_names:        
            if key == "left":
                self.mouse_x = -60
            elif key == "right":
                self.mouse_x = +60
            elif key == "up":
                self.mouse_y = -50
            elif key == "down":
                self.mouse_y = +50


def print_csgo_action(action: CSGOAction) -> Tuple[str]:
    action_names = [CSGO_KEYMAP[k] for k in action.keys] if len(action.keys) > 0 else []
    action_names = [x for x in action_names if not x.startswith("camera_")]
    keys   = " + ".join(action_names)     
    mouse  = str((action.mouse_x, action.mouse_y)) * (action.mouse_x != 0 or action.mouse_y != 0)
    clicks = "L" * action.l_click + " + " * (action.l_click and action.r_click) + "R" * action.r_click
    return keys, mouse, clicks     


MOUSE_X_POSSIBLES = [
    -1000,
    -500,
    -300,
    -200,
    -100,
    -60,
    -30,
    -20,
    -10,
    -4,
    -2,
    0,
    2,
    4,
    10,
    20,
    30,
    60,
    100,
    200,
    300,
    500,
    1000,
]

MOUSE_Y_POSSIBLES = [
    -200,
    -100,
    -50,
    -20,
    -10,
    -4,
    -2,
    0,
    2,
    4,
    10,
    20,
    50,
    100,
    200,
]

MOUSE_X_LIM = (MOUSE_X_POSSIBLES[0], MOUSE_X_POSSIBLES[-1])
MOUSE_Y_LIM = (MOUSE_Y_POSSIBLES[0], MOUSE_Y_POSSIBLES[-1])
N_KEYS = 11  # number of keyboard outputs, w,s,a,d,space,ctrl,shift,1,2,3,r
N_CLICKS = 2  # number of mouse buttons, left, right
N_MOUSE_X = len(MOUSE_X_POSSIBLES)  # number of outputs on mouse x axis
N_MOUSE_Y = len(MOUSE_Y_POSSIBLES)  # number of outputs on mouse y axis


def encode_csgo_action(csgo_action: CSGOAction, device: torch.device) -> torch.Tensor:

    # mouse_x = csgo_action.mouse_x
    # mouse_y = csgo_action.mouse_y

    keys_pressed_onehot = np.zeros(N_KEYS)
    mouse_x_onehot = np.zeros(N_MOUSE_X)
    mouse_y_onehot = np.zeros(N_MOUSE_Y)
    l_click_onehot = np.zeros(1)
    r_click_onehot = np.zeros(1)

    for key in csgo_action.key_names:
        if key == "w":
            keys_pressed_onehot[0] = 1
        elif key == "a":
            keys_pressed_onehot[1] = 1
        elif key == "s":
            keys_pressed_onehot[2] = 1
        elif key == "d":
            keys_pressed_onehot[3] = 1
        elif key == "space":
            keys_pressed_onehot[4] = 1
        elif key == "left ctrl":
            keys_pressed_onehot[5] = 1
        elif key == "left shift":
            keys_pressed_onehot[6] = 1
        elif key == "1":
            keys_pressed_onehot[7] = 1
        elif key == "2":
            keys_pressed_onehot[8] = 1
        elif key == "3":
            keys_pressed_onehot[9] = 1
        elif key == "r":
            keys_pressed_onehot[10] = 1

    l_click_onehot[0] = int(csgo_action.l_click)
    r_click_onehot[0] = int(csgo_action.r_click)

    mouse_x_onehot[MOUSE_X_POSSIBLES.index(csgo_action.mouse_x)] = 1
    mouse_y_onehot[MOUSE_Y_POSSIBLES.index(csgo_action.mouse_y)] = 1

    assert mouse_x_onehot.sum() == 1
    assert mouse_y_onehot.sum() == 1

    return torch.tensor(
        np.concatenate((
            keys_pressed_onehot,
            l_click_onehot,
            r_click_onehot,
            mouse_x_onehot,
            mouse_y_onehot,
        )),
        device=device,
        dtype=torch.float32,
    )
    

def decode_csgo_action(y_preds: torch.Tensor) -> CSGOAction:
    y_preds = y_preds.squeeze()
    keys_pred = y_preds[0:N_KEYS]
    l_click_pred = y_preds[N_KEYS : N_KEYS + 1]
    r_click_pred = y_preds[N_KEYS + 1 : N_KEYS + N_CLICKS]
    mouse_x_pred = y_preds[N_KEYS + N_CLICKS : N_KEYS + N_CLICKS + N_MOUSE_X]
    mouse_y_pred = y_preds[
        N_KEYS + N_CLICKS + N_MOUSE_X : N_KEYS + N_CLICKS + N_MOUSE_X + N_MOUSE_Y
    ]

    keys_pressed = []
    keys_pressed_onehot = np.round(keys_pred)
    if keys_pressed_onehot[0] == 1:
        keys_pressed.append("w")
    if keys_pressed_onehot[1] == 1:
        keys_pressed.append("a")
    if keys_pressed_onehot[2] == 1:
        keys_pressed.append("s")
    if keys_pressed_onehot[3] == 1:
        keys_pressed.append("d")
    if keys_pressed_onehot[4] == 1:
        keys_pressed.append("space")
    if keys_pressed_onehot[5] == 1:
        keys_pressed.append("left ctrl")
    if keys_pressed_onehot[6] == 1:
        keys_pressed.append("left shift")
    if keys_pressed_onehot[7] == 1:
        keys_pressed.append("1")
    if keys_pressed_onehot[8] == 1:
        keys_pressed.append("2")
    if keys_pressed_onehot[9] == 1:
        keys_pressed.append("3")
    if keys_pressed_onehot[10] == 1:
        keys_pressed.append("r")

    l_click = int(np.round(l_click_pred))
    r_click = int(np.round(r_click_pred))

    id = np.argmax(mouse_x_pred)
    mouse_x = MOUSE_X_POSSIBLES[id]
    id = np.argmax(mouse_y_pred)
    mouse_y = MOUSE_Y_POSSIBLES[id]

    keys_pressed = [pygame.key.key_code(x) for x in keys_pressed]

    return CSGOAction(keys_pressed, mouse_x, mouse_y, bool(l_click), bool(r_click))


def filter_keys_pressed_forbidden(keys_pressed: List[int], keymap: Dict[int, str] = CSGO_KEYMAP, forbidden_combinations: List[Set[str]] = CSGO_FORBIDDEN_COMBINATIONS) -> List[int]:
    keys = set()
    names = set()
    for key in keys_pressed:
        if key not in keymap:
            continue
        name = keymap[key]
        keys.add(key)
        names.add(name)
        for forbidden in forbidden_combinations:
            if forbidden.issubset(names):
                keys.remove(key)
                names.remove(name)
                break
    return list(filter(lambda key: key in keys, keys_pressed))
