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


def test_scenario20v20_debug_reward():
    env = make_scenario("identical_20_vs_20", debug_reward=True)
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


def run_test_state_obs_scenario(name):
    env = make_scenario(name, obs_type="vector")
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
    batch_size = 6
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


def run_test_scenario_vmapped(name):
    batch_size = 4
    env = make_scenario(name)
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


def run_test_with_map(name, obs_type):
    env = make_scenario(name, obs_type=obs_type, maps="all")
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
    state, obs, rewards, dones, ep_done = env.step(state, action, key_step)


def test_maps():
    run_test_with_map("hide_and_seek_5_vs_5", obs_type="vector")
    run_test_with_map("waypoint_5_vs_5", obs_type="vector")
    run_test_with_map("identical_20_vs_20", obs_type="rgb")


def test_scenario_other_scenarios():
    run_test_scenario_vmapped("hide_and_seek_5_vs_5")
    run_test_scenario_vmapped("waypoint_5_vs_5")
    run_test_state_obs_scenario("waypoint_5_vs_5")
    run_test_state_obs_scenario("hide_and_seek_5_vs_5")
    run_test_scenario_vmapped("waypoint_5_vs_3_fobs_rgb_maps_cont")
    run_test_state_obs_scenario("identical_5_vs_1_fobs_vec_void_cont")


if __name__ == "__main__":
    test_scenario20v20_debug_reward()
    test_scenario20v20()
    test_state_obs_scenario20v20()
    test_vmap_scenario20v20()