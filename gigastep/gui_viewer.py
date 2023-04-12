import math
import sys
import time

import cv2
import jax
import numpy as np
import jax.numpy as jnp
import pygame

class GigastepViewer:
    display = None

    def __init__(
        self,
        frame_size,
        show_num_agents=1,
    ):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.show_num_agents = show_num_agents
        self.image = None
        self.frame_size = frame_size
        self.should_quit = False
        self.should_reset = False
        self._should_pause = False
        if show_num_agents <= 6:
            self._num_cols = show_num_agents+1
            self._num_rows = 1
        else:
            self._num_cols = 6
            self._num_rows = math.ceil((show_num_agents+1) / self._num_cols)
        frame = (frame_size * self._num_cols, frame_size * self._num_rows)

        if GigastepViewer.display is None:
            GigastepViewer.display = pygame.display.set_mode(frame)

        self.action = jnp.array([0, 0, 0])

    @property
    def should_pause(self):
        p = self._should_pause
        self._should_pause = False
        return p

    def poll(self):
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        self.should_reset = False
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.should_quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                self._should_pause = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_BACKSPACE:
                self.should_reset = True

        action = [0, 0, 0]
        if keys[pygame.K_a]:
            action[0] = -1
        elif keys[pygame.K_d]:
            action[0] = 1
        if keys[pygame.K_w]:
            action[2] = 1
        elif keys[pygame.K_s]:
            action[2] = -1
        if keys[pygame.K_SPACE]:
            action[1] = 1
        elif keys[pygame.K_LSHIFT]:
            action[1] = -1
        if keys[pygame.K_UP]:
            action[2] = 1
        elif keys[pygame.K_DOWN]:
            action[2] = -1
        if keys[pygame.K_LEFT]:
            action[0] = -1
        elif keys[pygame.K_RIGHT]:
            action[0] = 1

        self.action = jnp.array(action)

    def draw(self, dyn, state, obs):
        frame_buffer = np.zeros(( self.frame_size * self._num_cols, self.frame_size * self._num_rows,3), dtype=np.uint8)


        global_obs = dyn.get_global_observation(state)
        global_obs = np.array(global_obs, dtype=np.uint8)
        # obs = cv2.cvtColor(np.array(obs), cv2.COLOR_RGB2BGR)
        global_obs = cv2.resize(
            global_obs,
            (self.frame_size, self.frame_size),
            interpolation=cv2.INTER_NEAREST,
        )
        frame_buffer[
            0 : self.frame_size,
            0 : self.frame_size,
        ] = global_obs

        for i in range(self.show_num_agents):
            obs_1 = obs[i]
            obs_1 = np.array(obs_1, dtype=np.uint8)
            # obs_1 = cv2.cvtColor(np.array(obs_1), cv2.COLOR_RGB2BGR)
            obs_1 = cv2.resize(
                obs_1,
                (self.frame_size, self.frame_size),
                interpolation=cv2.INTER_NEAREST,
            )
            row = (i+1) // self._num_cols
            col = (i+1) % self._num_cols
            frame_buffer[
                col * self.frame_size : (col + 1) * self.frame_size,
                row * self.frame_size : (row + 1) * self.frame_size,
            ] = obs_1
        self._show_image(frame_buffer)
        self.poll()
        self.clock.tick(60)
        return np.transpose(frame_buffer,[1,0,2])

    def _show_image(self, image):
        self.image = pygame.surfarray.make_surface(image)
        GigastepViewer.display.blit(self.image, (0, 0))
        pygame.display.update()

    def set_title(self, title):
        pygame.display.set_caption(title)

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