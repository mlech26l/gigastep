import os
import sys
import time

import cv2
import jax

from gigastep import GigastepViewer, GigastepEnv
from gigastep.scenarios import get_5v5_env, get_3v3_env, get_1v5_env, ScenarioBuilder

SLEEP_TIME = 0.01

def loop_env(env):
    viewer = GigastepViewer(84 * 2,show_num_agents=env.n_agents)
    rng = jax.random.PRNGKey(3)

    key, rng = jax.random.split(rng, 2)
    frame_idx = 0
    os.makedirs("video/scenario", exist_ok=True)
    while True:
        state, obs = env.reset(key)
        ep_done = False
        t = 0
        while not ep_done and t < 50:
            rng, key = jax.random.split(rng, 2)
            action = jax.random.uniform(key, shape=(env.n_agents, 3), minval=-1, maxval=1)
            rng, key = jax.random.split(rng, 2)
            state, obs, r, a, d = env.step(state, action, key)
            img = viewer.draw(env, state, obs)
            cv2.imwrite(f"video/scenario/frame_{frame_idx:05d}.png", img)
            if viewer.should_pause:
                return
            if viewer.should_quit:
                sys.exit(1)
            time.sleep(SLEEP_TIME)
            t += 1
            frame_idx += 1

def get_custom_scenario():
    builder = ScenarioBuilder()
    # 2 default, 1 tank, 1 sniper, 1 ranger
    builder.add_default_type(0)
    builder.add_default_type(0)
    builder.add_tank_type(0)
    builder.add_sniper_type(0)
    builder.add_ranger_type(0)

    builder.add_default_type(1)
    builder.add_default_type(1)
    builder.add_tank_type(1)
    builder.add_sniper_type(1)
    builder.add_ranger_type(1)
    builder.add_default_type(0)
    builder.add_default_type(0)
    builder.add_tank_type(0)
    builder.add_sniper_type(0)
    builder.add_ranger_type(0)

    builder.add_default_type(1)
    builder.add_default_type(1)

    return GigastepEnv(**builder.get_kwargs())

if __name__ == "__main__":
    # convert -delay 5 -loop 0 video/scenario/frame_*.png video/scenario.gif
    loop_env(env = get_custom_scenario())
    loop_env(env = get_5v5_env())
    loop_env(env = get_3v3_env())
    loop_env(env = get_1v5_env())