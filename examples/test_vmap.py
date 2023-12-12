import time

import jax
import jax.numpy as jnp
from gigastep import GigastepEnv
import time

if __name__ == "__main__":
    n_agents = 20
    env = GigastepEnv(n_agents=n_agents)
    batch_size = 5
    rng = jax.random.PRNGKey(1)
    rng, key = jax.random.split(rng, 2)
    key = jax.random.split(key, batch_size)
    obs, states = env.v_reset(key)
    t = 0
    while True:
        t += 1
        rng, key = jax.random.split(rng, 2)

        actions = jax.random.uniform(
            key, shape=(batch_size, n_agents, 3), minval=-1, maxval=1
        )
        rng, key = jax.random.split(rng, 2)
        key = jax.random.split(key, batch_size)
        # print("states.shape", states[0]["x"].shape)
        # print("actions.shape", actions.shape)
        obs, states, r, d, ep_dones = env.v_step(states, actions, key)

        print(f"t= {t}, ep_dones", ep_dones)
        # time.sleep(0.5)
        if jnp.any(ep_dones):
            print("resetting")
            rng, key = jax.random.split(rng, 2)
            obs, states = env.reset_done_episodes(obs, states, ep_dones, key)