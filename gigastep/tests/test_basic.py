# import pytest
from gigastep import make_scenario
import jax
import jax.numpy as jnp


def test_scenario20v20():
    env = make_scenario("identical_20_vs_20")
    rng = jax.random.PRNGKey(3)
    rng, key_reset = jax.random.split(rng, 2)

    ep_done = False
    state, obs = env.reset(key_reset)
    rng, key_action, key_step = jax.random.split(rng, 3)
    action = jax.random.uniform(
        key_action, shape=(env.n_agents, 3), minval=-1, maxval=1
    )
    state, obs, rewards, dones, ep_done = env.step(state, action, key_step)
    state, obs, rewards, dones, ep_done = env.step(state, action, key_step)


def test_state_obs_scenario20v20():
    env = make_scenario("identical_20_vs_20", obs_type="vector")
    rng = jax.random.PRNGKey(3)
    rng, key_reset = jax.random.split(rng, 2)

    ep_done = False
    state, obs = env.reset(key_reset)
    rng, key_action, key_step = jax.random.split(rng, 3)
    action = jax.random.uniform(
        key_action, shape=(env.n_agents, 3), minval=-1, maxval=1
    )
    state, obs, rewards, dones, ep_done = env.step(state, action, key_step)
    state, obs, rewards, dones, ep_done = env.step(state, action, key_step)


def test_no_waypoint_scenario20v20():
    env = make_scenario("identical_20_vs_20", obs_type="vector", enable_waypoints=False)
    rng = jax.random.PRNGKey(3)
    rng, key_reset = jax.random.split(rng, 2)

    ep_done = False
    state, obs = env.reset(key_reset)
    rng, key_action, key_step = jax.random.split(rng, 3)
    action = jax.random.uniform(
        key_action, shape=(env.n_agents, 3), minval=-1, maxval=1
    )
    state, obs, rewards, dones, ep_done = env.step(state, action, key_step)
    state, obs, rewards, dones, ep_done = env.step(state, action, key_step)


def test_vmap_scenario20v20():
    batch_size = 32
    env = make_scenario("identical_20_vs_20")
    rng = jax.random.PRNGKey(3)
    rng, key_reset = jax.random.split(rng, 2)
    key_reset = jax.random.split(key_reset, batch_size)

    state, obs = env.v_reset(key_reset)
    ep_dones = jnp.zeros(batch_size, dtype=jnp.bool_)
    rng, key_action, key_step = jax.random.split(rng, 3)
    action = jax.random.uniform(
        key_action, shape=(batch_size, env.n_agents, 3), minval=-1, maxval=1
    )
    key_step = jax.random.split(key_step, batch_size)
    state, obs, rewards, dones, ep_dones = env.v_step(state, action, key_step)
    state, obs, rewards, dones, ep_dones = env.v_step(state, action, key_step)
    state, obs, rewards, dones, ep_dones = env.v_step(state, action, key_step)
    # obs is an uint8 array of shape [batch_size, n_agents, 84,84,3]
    # rewards is a float32 array of shape [batch_size, n_agents]
    # dones is a bool array of shape [batch_size, n_agents]
    # ep_done is a bool array of shape [batch_size]

    # In case at least one episode is done, reset the state of the done episodes only
    rng, key = jax.random.split(rng, 2)
    states, obs = env.reset_done_episodes(state, obs, ep_dones, key)


if __name__ == "__main__":
    test_scenario20v20()
    test_state_obs_scenario20v20()
    test_vmap_scenario20v20()