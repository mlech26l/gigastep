import os
import sys
import time

import cv2
import jax
import numpy as np

from gigastep import GigastepViewer, GigastepEnv, make_scenario, ScenarioBuilder
from gigastep.evaluator import Evaluator
import jax.numpy as jnp

SLEEP_TIME = 0.01


def loop_env(env):
    evaluator = Evaluator(env)
    viewer = GigastepViewer(84 * 2, show_num_agents=env.n_agents)
    rng = jax.random.PRNGKey(3)

    for opponent in evaluator.policies:
        print("Opponent", str(opponent))
        for ep_idx in range(5):
            ep_done = False
            key, rng = jax.random.split(rng, 2)
            state, obs = env.reset(key)
            while not ep_done:
                rng, key, key2 = jax.random.split(rng, 3)
                action_ego = jnp.zeros((env.n_agents, 3))  # ego does nothing
                action_opp = opponent.apply(obs, key2)

                action = evaluator.merge_actions(action_ego, action_opp)

                rng, key = jax.random.split(rng, 2)
                state, obs, r, dones, ep_done = env.step(state, action, key)
                evaluator.update_step(r, dones, ep_done)

                img = viewer.draw(env, state, obs)
                if viewer.should_pause:
                    while True:
                        img = viewer.draw(env, state, obs)
                        time.sleep(SLEEP_TIME)
                        if viewer.should_pause:
                            break
                if viewer.should_quit:
                    sys.exit(1)
                time.sleep(SLEEP_TIME)
            evaluator.update_episode()
            print(str(evaluator))
            # if frame_idx > 400:
            #     sys.exit(1)


def loop_env_vectorized(env):
    evaluator = Evaluator(env)
    batch_size = 20
    rng = jax.random.PRNGKey(3)

    for opponent in evaluator.policies:
        print("Opponent", str(opponent))
        for ep_idx in range(5):
            ep_done = np.zeros(batch_size, dtype=jnp.bool_)
            key, rng = jax.random.split(rng, 2)
            key = jax.random.split(key, batch_size)
            state, obs = env.v_reset(key)
            while not jnp.all(ep_done):
                rng, key, key2 = jax.random.split(rng, 3)
                action_ego = jnp.zeros(
                    (batch_size, env.n_agents, 3)
                )  # ego does nothing
                key2 = jax.random.split(key2, batch_size)
                action_opp = opponent.v_apply(obs, key2)

                action = evaluator.v_merge_actions(action_ego, action_opp)

                rng, key = jax.random.split(rng, 2)
                key = jax.random.split(key, batch_size)
                state, obs, r, dones, ep_done = env.v_step(state, action, key)
                evaluator.update_step(r, dones, ep_done)

                time.sleep(SLEEP_TIME)
            evaluator.update_episode()
            print(str(evaluator))
            # if frame_idx > 400:
            #     sys.exit(1)


if __name__ == "__main__":
    # convert -delay 3 -loop 0 video/scenario/frame_*.png video/scenario.webp
    loop_env_vectorized(
        env=make_scenario("identical_20_vs_20", use_stochastic_comm=False)
    )
    loop_env(env=make_scenario("identical_20_vs_20", use_stochastic_comm=False))
    # loop_env(env = make_scenario("identical_20_vs_20"))
    # loop_env(env = make_scenario("identical_20_vs_20"))