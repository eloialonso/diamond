import pygame


CSGO_KEYMAP = {
    pygame.K_w: "up",
    pygame.K_d: "right",
    pygame.K_a: "left",
    pygame.K_s: "down",
    pygame.K_SPACE: "jump",
    pygame.K_LCTRL: "crouch",
    pygame.K_LSHIFT: "walk",
    pygame.K_1: "weapon1",
    pygame.K_2: "weapon2",
    pygame.K_3: "weapon3",
    pygame.K_r: "reload",

    # Override mouse movement with arrows
    pygame.K_UP: "camera_up",
    pygame.K_RIGHT: "camera_right",
    pygame.K_LEFT: "camera_left",
    pygame.K_DOWN: "camera_down",
}


CSGO_FORBIDDEN_COMBINATIONS = [
    {"up", "down"},
    {"left", "right"},
    {"weapon1", "weapon2"},
    {"weapon1", "weapon3"},
    {"weapon2", "weapon3"},
    {"camera_up", "camera_down"},
    {"camera_left", "camera_right"},
]
