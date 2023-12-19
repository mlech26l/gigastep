import os
import sys
import time

import cv2
import jax
import jax.numpy as jnp
from gigastep import GigastepViewer, GigastepEnv, make_scenario, ScenarioBuilder

SLEEP_TIME = 0.01


def loop_env(env):
    viewer = GigastepViewer(84 * 2, show_num_agents=env.n_agents)
    rng = jax.random.PRNGKey(3)

    key, rng = jax.random.split(rng, 2)
    frame_idx = 0
    os.makedirs("video/scenario", exist_ok=True)
    while True:
        obs, state = env.reset(key)
        ep_done = False
        t = 0
        # while not ep_done and t < 50:
        while True:
            rng, key = jax.random.split(rng, 2)
            action = jax.random.uniform(
                key, shape=(env.n_agents, 3), minval=-1, maxval=1
            )
            # action = jnp.zeros((env.n_agents, 3))
            rng, key = jax.random.split(rng, 2)
            obs, state, r, a, d = env.step(state, action, key)
            img = viewer.draw(env, state, obs)

            img = img[0 : 2 * img.shape[0] // 3]
            cv2.imwrite(f"video/scenario/frame_{frame_idx:04d}.png", img)
            if viewer.should_pause:
                while True:
                    img = viewer.draw(env, state, obs)
                    time.sleep(SLEEP_TIME)
                    if viewer.should_pause:
                        break
            if viewer.should_reset:
                break
            if viewer.should_quit:
                sys.exit(1)
            time.sleep(SLEEP_TIME)
            t += 1
            frame_idx += 1
        # if frame_idx > 400:
        #     sys.exit(1)


if __name__ == "__main__":
    # convert -delay 3 -loop 0 video/scenario/frame_*.png video/scenario.webp
    loop_env(env=make_scenario("waypoint_5_vs_5", use_stochastic_comm=False))
    # loop_env(env=make_scenario("special_20_vs_20", use_stochastic_comm=False))
    # loop_env(env = make_scenario("identical_20_vs_20"))
    # loop_env(env = make_scenario("identical_20_vs_20"))