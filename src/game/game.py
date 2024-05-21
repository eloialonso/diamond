from typing import Tuple, Union

import numpy as np
import pygame
from PIL import Image

from .dataset_env import DatasetEnv
from .keymap import Keymap
from .play_env import PlayEnv


class Game:
    def __init__(
        self,
        play_env: Union[PlayEnv, DatasetEnv],
        keymap: Keymap,
        size: Tuple[int, int],
        fps: int,
        verbose: bool,
    ) -> None:
        self.env = play_env
        self.keymap = keymap
        self.height, self.width = size
        self.fps = fps
        self.verbose = verbose

        print("\nControls (general):\n")
        print("⏎ : reset env")
        print(". : pause/unpause")
        print("e : step-by-step (when paused)")
        self.env.print_controls()
        print("\n")

    def run(self) -> None:
        pygame.init()

        header_height = 150 if self.verbose else 0
        font_size = 16
        screen = pygame.display.set_mode((self.width, self.height + header_height))
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("mono", font_size)
        header_rect = pygame.Rect(0, 0, self.width, header_height)

        def clear_header():
            pygame.draw.rect(screen, pygame.Color("black"), header_rect)
            pygame.draw.rect(screen, pygame.Color("white"), header_rect, 1)

        def draw_text(text, idx_line, idx_column, num_cols):
            pos = (5 + idx_column * int(self.width // num_cols), 5 + idx_line * font_size)
            assert (0 <= pos[0] <= self.width) and (0 <= pos[1] <= header_height)
            screen.blit(font.render(text, True, pygame.Color("white")), pos)

        def draw_game(obs):
            assert obs.ndim == 4 and obs.size(0) == 1
            img = Image.fromarray(obs[0].add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy())
            pygame_image = np.array(img.resize((self.width, self.height), resample=Image.NEAREST)).transpose((1, 0, 2))
            surface = pygame.surfarray.make_surface(pygame_image)
            screen.blit(surface, (0, header_height))

        def reset():
            nonlocal obs, info, do_reset, ep_return, ep_length
            obs, info = self.env.reset()
            do_reset = False
            ep_return = 0
            ep_length = 0

        obs, info, do_reset, ep_return, ep_length = (None,) * 5

        reset()
        do_wait = False
        should_stop = False

        while not should_stop:
            do_one_step = False
            action = 0  # noop
            pygame.event.pump()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    should_stop = True

                if event.type != pygame.KEYDOWN:
                    continue

                if event.key == pygame.K_RETURN:
                    do_reset = True

                if event.key == pygame.K_PERIOD:
                    do_wait = not do_wait
                    print("Game paused." if do_wait else "Game resumed.")

                if event.key == pygame.K_e:
                    do_one_step = True

                if event.key == pygame.K_m:
                    do_reset = self.env.next_mode()

                if event.key == pygame.K_UP:
                    do_reset = self.env.next_axis_1()

                if event.key == pygame.K_DOWN:
                    do_reset = self.env.prev_axis_1()

                if event.key == pygame.K_RIGHT:
                    do_reset = self.env.next_axis_2()

                if event.key == pygame.K_LEFT:
                    do_reset = self.env.prev_axis_2()

                if event.key in self.keymap.keys():
                    action = self.keymap[event.key]

            if action == 0:
                pressed = pygame.key.get_pressed()
                for key, action in self.keymap.items():
                    if pressed[key]:
                        break
                else:
                    action = 0

            if do_reset:
                reset()

            if do_wait and not do_one_step:
                continue

            next_obs, rew, end, trunc, info = self.env.step(action)

            ep_return += rew.item()
            ep_length += 1

            draw_game(obs)

            if self.verbose and info is not None:
                clear_header()
                assert isinstance(info, dict) and "header" in info
                header = info["header"]
                num_cols = len(header)
                for j, col in enumerate(header):
                    for i, row in enumerate(col):
                        draw_text(row, idx_line=i, idx_column=j, num_cols=num_cols)

            pygame.display.flip()  # update screen
            clock.tick(self.fps)  # ensures game maintains the given frame rate

            if end or trunc:
                reset()

            else:
                obs = next_obs

        pygame.quit()
