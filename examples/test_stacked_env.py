import time

import jax
import jax.numpy as jnp
from gigastep import GigastepEnv, EnvFrameStack


if __name__ == "__main__":
    n_agents = 20
    env = GigastepEnv(n_agents=n_agents)
    env_stack = EnvFrameStack(env, nstack = 3)

    batch_size = 5
    rng = jax.random.PRNGKey(1)
    rng, key = jax.random.split(rng, 2)
    key = jax.random.split(key, batch_size)
    obs, states = env_stack.v_reset(key)
    t = 0
    while True:
        t += 1
        rng, key = jax.random.split(rng, 2)

        actions = jax.random.uniform(
            key, shape=(batch_size, n_agents, 3), minval=-1, maxval=1
        )
        rng, key = jax.random.split(rng, 2)
        key = jax.random.split(key, batch_size)

        obs, states, r, d, ep_dones = env_stack.v_step(states, actions, key)

        print(f"t= {t}, ep_dones", ep_dones)
        # time.sleep(0.5)
        if jnp.any(ep_dones):
            print("resetting")
            rng, key = jax.random.split(rng, 2)
            obs, states = env_stack.reset_done_episodes(obs, states, ep_dones, key)
