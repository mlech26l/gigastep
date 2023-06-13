import math
import sys
import time

import jax
import numpy as np
import jax.numpy as jnp
import importlib

from gigastep.joystick_input import JoystickInput


def discretize(x, threshold=0.3):
    if x >= threshold:
        return 1
    elif x <= -threshold:
        return 2
    return 0


class GigastepViewer:
    display = None

    def __init__(
        self, frame_size, show_global_state=True, show_num_agents=0, headless=False
    ):
        # only import pygame if the viewer is used
        try:
            self.pygame = importlib.import_module("pygame")
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install pygame (``pip3 install pygame``) to use the viewer"
            )
        try:
            self.cv2 = importlib.import_module("cv2")
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install cv2 (``pip3 install opencv-python``) to use the viewer"
            )
        self.pygame.init()
        self.clock = self.pygame.time.Clock()
        self.show_num_agents = show_num_agents
        self.show_global_state = show_global_state
        self.image = None
        self.frame_size = frame_size
        self.should_quit = False
        self.should_reset = False
        self._should_pause = False
        self.headless = headless
        if show_num_agents <= 6:
            self._num_cols = show_num_agents + int(show_global_state)
            self._num_rows = 1
        else:
            self._num_cols = 6
            self._num_rows = math.ceil(
                (show_num_agents + int(show_global_state)) / self._num_cols
            )
        frame = (frame_size * self._num_cols, frame_size * self._num_rows)

        if GigastepViewer.display is None and self.headless is False:
            GigastepViewer.display = self.pygame.display.set_mode(frame)

        self.discrete_action = 0
        self.continuous_action = np.zeros(3)
        self.joystick = JoystickInput(self.pygame)

    @property
    def should_pause(self):
        p = self._should_pause
        self._should_pause = False
        return p

    def poll(self):
        self.pygame.event.pump()
        keys = self.pygame.key.get_pressed()

        self.should_reset = False
        for event in self.pygame.event.get():
            if event.type == self.pygame.KEYDOWN and event.key == self.pygame.K_ESCAPE:
                self.should_quit = True
            elif (
                event.type == self.pygame.KEYDOWN and event.key == self.pygame.K_RETURN
            ):
                self._should_pause = True
            elif (
                event.type == self.pygame.KEYDOWN
                and event.key == self.pygame.K_BACKSPACE
            ):
                self.should_reset = True

        action = np.zeros(3)
        if self.joystick.is_connected():
            action, pause, reset, quit = self.joystick.poll()
            if pause:
                self._should_pause = True
            if reset:
                self.should_reset = True
            if quit:
                self.should_quit = True

        if keys[self.pygame.K_a]:
            action[0] = -1
        elif keys[self.pygame.K_d]:
            action[0] = 1
        if keys[self.pygame.K_w]:
            action[2] = 1
        elif keys[self.pygame.K_s]:
            action[2] = -1
        if keys[self.pygame.K_SPACE]:
            action[1] = 1
        elif keys[self.pygame.K_LSHIFT]:
            action[1] = -1
        if keys[self.pygame.K_UP]:
            action[2] = 1
        elif keys[self.pygame.K_DOWN]:
            action[2] = -1
        if keys[self.pygame.K_LEFT]:
            action[0] = -1
        elif keys[self.pygame.K_RIGHT]:
            action[0] = 1

        action_id = discretize(action[0])
        action_id += 3 * discretize(action[1])
        action_id += 9 * discretize(action[2])
        self.discrete_action = action_id
        self.continuous_action = jnp.array(action)

    def draw(self, dyn, state, obs):
        frame_buffer = np.zeros(
            (self.frame_size * self._num_cols, self.frame_size * self._num_rows, 3),
            dtype=np.uint8,
        )

        if self.show_global_state:
            global_obs = dyn.get_global_observation(state)
            global_obs = np.array(global_obs, dtype=np.uint8)
            # obs = self.cv2.cvtColor(np.array(obs), self.cv2.COLOR_RGB2BGR)
            global_obs = self.cv2.resize(
                global_obs,
                (self.frame_size, self.frame_size),
                interpolation=self.cv2.INTER_NEAREST,
            )
            frame_buffer[
                0 : self.frame_size,
                0 : self.frame_size,
            ] = global_obs

        for i in range(self.show_num_agents):
            obs_1 = obs[i]
            obs_1 = np.array(obs_1, dtype=np.uint8)
            obs_1 = np.maximum(obs_1, 255 * (1 - state[0]["alive"][i]))
            # obs_1 = self.cv2.cvtColor(np.array(obs_1), self.cv2.COLOR_RGB2BGR)
            obs_1 = self.cv2.resize(
                obs_1,
                (self.frame_size, self.frame_size),
                interpolation=self.cv2.INTER_NEAREST,
            )
            ## Code to rotate the image to match the heading of the agent upwards
            ## This is confusion to play because outward things move very fast but may be useful later
            # heading = 180 * float(state[0]["heading"][0]) / np.pi + 90
            # M = self.cv2.getRotationMatrix2D(
            #     (obs_1.shape[0] // 2, obs_1.shape[1] // 2),
            #     -heading,
            #     1.0,
            # )
            # obs_1 = self.cv2.warpAffine(obs_1, M, (obs_1.shape[0], obs_1.shape[1]))

            idx = i + int(self.show_global_state)
            row = idx // self._num_cols
            col = idx % self._num_cols
            frame_buffer[
                col * self.frame_size : (col + 1) * self.frame_size,
                row * self.frame_size : (row + 1) * self.frame_size,
            ] = obs_1
        if not self.headless:
            self._show_image(frame_buffer)
        self.poll()
        self.clock.tick(10)
        frame_buffer = self.cv2.cvtColor(frame_buffer, self.cv2.COLOR_RGB2BGR)
        return np.transpose(frame_buffer, [1, 0, 2])

    def _show_image(self, image):
        self.image = self.pygame.surfarray.make_surface(image)
        GigastepViewer.display.blit(self.image, (0, 0))
        self.pygame.display.update()

    def set_title(self, title):
        self.pygame.display.set_caption(title)

    # def start(self):
    #     running = True
    #     while running:
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 running = False
    #
    #         self.display.blit(self.image, (0, 0))
    #         pygame.display.update()
    #
    #