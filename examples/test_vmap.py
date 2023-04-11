import time

import jax
import jax.numpy as jnp
from gigastep import GigastepEnv
import time

if __name__ == '__main__':

    n_agents = 20
    dyn = GigastepEnv(n_agents=n_agents)
    batch_size = 32
    rng = jax.random.PRNGKey(1)
    rng, key = jax.random.split(rng, 2)
    key = jax.random.split(key, batch_size)
    while True:
        states, obs = dyn.v_reset(key)
        while True:
            rng, key = jax.random.split(rng, 2)

            actions = jax.random.uniform(key, shape=(batch_size,n_agents, 3), minval=-1, maxval=1)
            rng, key = jax.random.split(rng, 2)
            key = jax.random.split(key, batch_size)
            print("states.shape", states[0]["x"].shape)
            print("actions.shape", actions.shape)
            states, obs, r, a, d = dyn.v_step(states, actions, key)

            time.sleep(0.5)