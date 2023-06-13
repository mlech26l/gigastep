import jax

import time
import jax.numpy as jnp

from gigastep import GigastepEnv, stack_agents, GigastepViewer, make_scenario


def loop_user():
    viewer = GigastepViewer(84 * 4, show_num_agents=1)
    viewer.set_title("User input")
    env = make_scenario("identical_5_vs_5", discrete_actions=False)
    rng = jax.random.PRNGKey(1)
    while True:
        key, rng = jax.random.split(rng, 2)
        state, obs = env.reset(key)
        ep_done = False
        t = 0
        while not ep_done and t < 50:
            rng, key = jax.random.split(rng, 2)
            a1 = viewer.continuous_action
            a2 = jax.random.uniform(key, shape=(env.n_agents, 3), minval=-1, maxval=1)
            action = jnp.where(jnp.arange(env.n_agents)[:, None] == 0, a1, a2)
            rng, key = jax.random.split(rng, 2)
            state, obs, r, a, ep_done = env.step(state, action, key)
            viewer.draw(env, state, obs)
            if viewer.should_reset:
                break
            if viewer.should_quit:
                return


if __name__ == "__main__":
    loop_user()