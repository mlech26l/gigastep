import sys
import time

import jax
import jax.numpy as jnp
from gigastep import GigastepEnv, stack_agents
from gigastep import GigastepViewer


SLEEP_TIME = 0.01


def loop_random_agents():
    viewer = GigastepViewer(120 * 4,show_agent1=False)
    viewer.set_title("10 random agents")
    n_agents = 2000
    resolution = 160
    dyn = GigastepEnv(n_agents=n_agents,limit_x=40,limit_y=40,resolution_x=resolution,resolution_y=resolution)
    rng = jax.random.PRNGKey(3)

    key, rng = jax.random.split(rng, 2)
    while True:
        state, obs = dyn.reset(key)
        while jnp.sum(dyn.get_dones(state)) > 0:
            rng, key = jax.random.split(rng, 2)
            action = jax.random.uniform(key, shape=(n_agents, 3), minval=-1, maxval=1)
            rng, key = jax.random.split(rng, 2)
            state, obs, r, a, d = dyn.step(state, action, key)
            viewer.draw(dyn, state, obs)
            if viewer.should_pause:
                return
            if viewer.should_quit:
                sys.exit(1)
            time.sleep(SLEEP_TIME)


if __name__ == "__main__":
    loop_random_agents()