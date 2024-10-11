from typing import Tuple, Union

import numpy as np
import pygame
from PIL import Image

from csgo.action_processing import CSGOAction
from .dataset_env import DatasetEnv
from .play_env import PlayEnv


class Game:
    def __init__(
        self,
        play_env: Union[PlayEnv, DatasetEnv],
        size: Tuple[int, int],
        mouse_multiplier: int,
        fps: int,
        verbose: bool,
    ) -> None:
        self.env = play_env
        self.height, self.width = size
        self.mouse_multiplier = mouse_multiplier
        self.fps = fps
        self.verbose = verbose

        print("\nControls:\n")
        print("Esc : quit")
        print(" ⏎  : reset env")
        print(" .  : pause/unpause")
        print(" e  : step-by-step (when paused)")
        self.env.print_controls()
        print("\n")
        input("Press enter to start")

    def run(self) -> None:
        pygame.init()

        header_height = 150 if self.verbose else 0
        header_width = 540
        font_size = 16
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("mono", font_size)
        x_center, y_center = screen.get_rect().center
        x_header = x_center - header_width // 2
        y_header = y_center - self.height // 2 - header_height - 10
        header_rect = pygame.Rect(x_header, y_header, header_width, header_height)

        def clear_header():
            pygame.draw.rect(screen, pygame.Color("black"), header_rect)
            pygame.draw.rect(screen, pygame.Color("white"), header_rect, 1)

        def draw_text(text, idx_line, idx_column, num_cols):
            x_pos = 5 + idx_column * int(header_width // num_cols)
            y_pos = 5 + idx_line * font_size
            assert (0 <= x_pos <= header_width) and (0 <= y_pos <= header_height)
            screen.blit(font.render(text, True, pygame.Color("white")), (x_header + x_pos, y_header + y_pos))
 
        def draw_obs(obs, obs_low_res=None):
            assert obs.ndim == 4 and obs.size(0) == 1
            img = Image.fromarray(obs[0].add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy())
            pygame_image = np.array(img.resize((self.width, self.height), resample=Image.BICUBIC)).transpose((1, 0, 2))
            surface = pygame.surfarray.make_surface(pygame_image)
            screen.blit(surface, (x_center - self.width // 2, y_center - self.height // 2))

            if obs_low_res is not None:
                assert obs_low_res.ndim == 4 and obs_low_res.size(0) == 1
                img = Image.fromarray(obs_low_res[0].add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy())
                h = self.height * obs_low_res.size(2) // obs.size(2)
                w = self.width * obs_low_res.size(3) // obs.size(3)
                pygame_image = np.array(img.resize((w, h), resample=Image.BICUBIC)).transpose((1, 0, 2))
                surface = pygame.surfarray.make_surface(pygame_image)
                screen.blit(surface, (x_header + header_width - w - 5, y_header + 5 + font_size))
                # screen.blit(surface, (x_center - w // 2, y_center + self.height // 2))

        def reset():
            nonlocal obs, info, do_reset, ep_return, ep_length, keys_pressed, l_click, r_click
            obs, info = self.env.reset()
            pygame.event.clear()
            do_reset = False
            ep_return = 0
            ep_length = 0
            keys_pressed = []
            l_click = r_click = False

        obs, info, do_reset, ep_return, ep_length, keys_pressed, l_click, r_click = (None,) * 8

        reset()
        do_wait = False
        should_stop = False

        while not should_stop:
            do_one_step = False
            mouse_x, mouse_y = 0, 0
            pygame.event.pump()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    should_stop = True

                if event.type == pygame.MOUSEMOTION:
                    mouse_x, mouse_y = event.rel
                    mouse_x *= self.mouse_multiplier
                    mouse_y *= self.mouse_multiplier

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        l_click = True
                    if event.button == 3:
                        r_click = True

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        l_click = False
                    if event.button == 3:
                        r_click = False

                if event.type == pygame.KEYDOWN:
                    keys_pressed.append(event.key)
                
                elif event.type == pygame.KEYUP and event.key in keys_pressed:
                    keys_pressed.remove(event.key)

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

            if do_reset:
                reset()

            if do_wait and not do_one_step:
                continue

            csgo_action = CSGOAction(keys_pressed, mouse_x, mouse_y, l_click, r_click)
            next_obs, rew, end, trunc, info = self.env.step(csgo_action)
            
            ep_return += rew.item()
            ep_length += 1

            if self.verbose and info is not None:
                clear_header()
                assert isinstance(info, dict) and "header" in info
                header = info["header"]
                num_cols = len(header)
                for j, col in enumerate(header):
                    for i, row in enumerate(col):
                        draw_text(row, idx_line=i, idx_column=j, num_cols=num_cols)

            draw_low_res = self.verbose and "obs_low_res" in info and self.width == 280
            if draw_low_res:
                draw_obs(obs, info["obs_low_res"])
                draw_text("  Pre-upsampling:", 0, 2, 3)
            else:
                draw_obs(obs, None)

            pygame.display.flip()  # update screen
            clock.tick(self.fps)  # ensures game maintains the given frame rate

            if end or trunc:
                reset()

            else:
                obs = next_obs

        pygame.quit()
