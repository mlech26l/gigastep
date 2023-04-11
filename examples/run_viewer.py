import jax

import time
import jax.numpy as jnp

from gigastep import GigastepEnv, stack_agents, GigastepViewer


def loop_user():
    viewer = GigastepViewer(84 * 4)
    viewer.set_title("User input")
    dyn = GigastepEnv()
    rng = jax.random.PRNGKey(1)
    while True:
        s1 = GigastepEnv.get_initial_state(x=3, y=3, team=1)
        s2 = GigastepEnv.get_initial_state(x=6, y=6, team=1)
        # state = jnp.stack([s1, s2, s3], axis=0)
        state = stack_agents(s1, s2)
        t = 0
        while jnp.sum(dyn.get_dones(state)) > 0:
            if viewer.should_reset:
                s1 = GigastepEnv.get_initial_state(x=3, y=3, team=1)
                s2 = GigastepEnv.get_initial_state(x=6, y=6, team=1)
                state = stack_agents(s1, s2)

            rng, key = jax.random.split(rng, 2)
            a1 = viewer.action
            a2 = jax.random.uniform(key, shape=(3,), minval=-1, maxval=1)
            action = jnp.stack([a1, a2], axis=0)
            rng, key = jax.random.split(rng, 2)
            state, obs, r, a, d = dyn.step(state, action, key)
            viewer.draw(dyn, state, obs)
            if viewer.should_quit:
                return


if __name__ == "__main__":
    loop_user()