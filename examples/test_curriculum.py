from gigastep import make_scenario
import jax

env = make_scenario("identical_20_vs_20")
rng = jax.random.PRNGKey(3)
rng, key_reset = jax.random.split(rng, 2)

ep_done = False
state, obs = env.reset(key_reset)
state = env.set_aux_reward_factor(state, 0.5)
while not ep_done:
    rng, key_action, key_step = jax.random.split(rng, 3)
    action = jax.random.uniform(
        key_action, shape=(env.n_agents, 3), minval=-1, maxval=1
    )
    state, obs, rewards, dones, ep_done = env.step(state, action, key_step)
    # obs is an uint8 array of shape [n_agents, 84,84,3]
    # rewards is a float32 array of shape [n_agents]
    # dones is a bool array of shape [n_agents]
    # ep_done is a bool


from gigastep import make_scenario
import jax
import jax.numpy as jnp

batch_size = 32
env = make_scenario("identical_20_vs_20")
rng = jax.random.PRNGKey(3)
rng, key_reset = jax.random.split(rng, 2)
key_reset = jax.random.split(key_reset, batch_size)

state, obs = env.v_reset(key_reset)
state = env.v_set_aux_reward_factor(state, 0.5)

ep_dones = jnp.zeros(batch_size, dtype=jnp.bool_)
while not jnp.all(ep_dones):
    rng, key_action, key_step = jax.random.split(rng, 3)
    action = jax.random.uniform(
        key_action, shape=(batch_size, env.n_agents, 3), minval=-1, maxval=1
    )
    key_step = jax.random.split(key_step, batch_size)
    state, obs, rewards, dones, ep_dones = env.v_step(state, action, key_step)
    # obs is an uint8 array of shape [batch_size, n_agents, 84,84,3]
    # rewards is a float32 array of shape [batch_size, n_agents]
    # dones is a bool array of shape [batch_size, n_agents]
    # ep_done is a bool array of shape [batch_size]

    # In case at least one episode is done, reset the state of the done episodes only
    if jnp.any(ep_dones):
        rng, key = jax.random.split(rng, 2)
        states, obs = env.reset_done_episodes(state, obs, ep_dones, key)