import time
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from gigastep.torch import torch2jax, jax2torch

import jax
import jax.numpy as jnp
from gigastep import GigastepEnv, make_scenario
import time
import torch

if __name__ == "__main__":
    env = make_scenario("identical_20_vs_20", obs_type="rgb")
    batch_size = 5

    rng = jax.random.PRNGKey(1)
    rng, key = jax.random.split(rng, 2)
    key = jax.random.split(key, batch_size)
    obs, states = env.v_reset(key)
    t = 0
    while True:
        t += 1
        rng, key = jax.random.split(rng, 2)

        # Create action with torch and move to JAX
        action = torch.ones((batch_size, env.n_agents, 3))
        action = action.to("cuda")
        action = torch2jax(action)

        key = jax.random.split(key, batch_size)
        obs, states, r, d, ep_dones = env.v_step(states, action, key)

        # Move obs to torch
        torch_obs = jax2torch(obs)
        print("obs.shape", torch_obs.shape)
        print("obs.dtype", torch_obs.dtype)
        print("obs.device", torch_obs.device)

        print(f"t= {t}, ep_dones", ep_dones)
        # time.sleep(0.5)
        if jnp.any(ep_dones):
            print("resetting")
            rng, key = jax.random.split(rng, 2)
            obs, states = env.reset_done_episodes(obs, states, ep_dones, key)