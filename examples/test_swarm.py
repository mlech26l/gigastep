import os
import sys
import time

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from gigastep import GigastepEnv, stack_agents
from gigastep import GigastepViewer


SLEEP_TIME = 0.01


def save_frame(
    dyn, state, step_obs, filename, frame_size=4 * 84, obs1=True, obs2=False
):
    obs = dyn.get_global_observation(state)
    obs = np.array(obs, dtype=np.uint8)
    # obs = cv2.cvtColor(np.array(obs), cv2.COLOR_RGB2BGR)
    obs = cv2.resize(obs, (frame_size, frame_size), interpolation=cv2.INTER_NEAREST)

    image = obs
    if obs1:
        obs_1 = step_obs[0]
        obs_1 = np.array(obs_1, dtype=np.uint8)
        # obs_1 = cv2.cvtColor(np.array(obs_1), cv2.COLOR_RGB2BGR)
        obs_1 = cv2.resize(
            obs_1, (frame_size, frame_size), interpolation=cv2.INTER_NEAREST
        )
        image = np.concatenate([image, obs_1], axis=1)
    if obs2:
        obs_2 = step_obs[1]
        obs_2 = np.array(obs_2, dtype=np.uint8)
        obs_2 = cv2.resize(
            obs_2, (frame_size, frame_size), interpolation=cv2.INTER_NEAREST
        )
        image = np.concatenate([image, obs_2], axis=1)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image)


def loop_random_agents():
    viewer = GigastepViewer(120 * 4, show_num_agents=0)
    viewer.set_title("10 random agents")
    n_agents = 2000
    resolution = 160
    dyn = GigastepEnv(
        n_agents=n_agents,
        limit_x=40,
        limit_y=40,
        resolution_x=resolution,
        resolution_y=resolution,
    )
    rng = jax.random.PRNGKey(3)
    os.makedirs("video/1000agents", exist_ok=True)
    key, rng = jax.random.split(rng, 2)
    frameid = 0
    while True:
        obs, state = dyn.reset(key)
        while jnp.sum(dyn.get_dones(state)) > 0:
            print(frameid)
            rng, key = jax.random.split(rng, 2)
            action = jax.random.uniform(key, shape=(n_agents, 3), minval=-1, maxval=1)
            rng, key = jax.random.split(rng, 2)
            obs, state, r, a, d = dyn.step(state, action, key)
            frame = viewer.draw(dyn, state, obs)
            save_frame(
                dyn, state, obs, f"video/1000agents/frame_{frameid:04d}.png", obs1=True
            )
            if viewer.should_pause:
                return
            if viewer.should_quit:
                sys.exit(1)
            time.sleep(SLEEP_TIME)
            frameid += 1


if __name__ == "__main__":
    loop_random_agents()