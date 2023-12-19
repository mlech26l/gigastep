from gigastep import make_scenario
import jax

env = make_scenario("identical_20_vs_20", obs_type="vector")
rng = jax.random.PRNGKey(3)
rng, key_reset = jax.random.split(rng, 2)

ep_done = False
obs, state = env.reset(key_reset)
t = 0
while not ep_done:
    rng, key_action, key_step = jax.random.split(rng, 3)
    action = jax.random.uniform(
        key_action, shape=(env.n_agents, 3), minval=-1, maxval=1
    )
    obs, state, rewards, dones, ep_done = env.step(state, action, key_step)
    if t <= 1:
        print("obs.shape", obs.shape)
        for i in range(20):
            print(f"obs[0]->{i}", obs[0, 6 * i : 6 * i + 6])

        print(f"ego.x={state[0]['x'][0]}," f"ego.y={state[0]['y'][0]}")
        print("-----------------------------------")
        for i in range(20):
            print(f"obs[0]->{i}", obs[1, 6 * i : 6 * i + 6])

        print(f"ego.x={state[0]['x'][1]}," f"ego.y={state[0]['y'][1]}")
    t += 1


import jax.numpy as jnp

batch_size = 32
env = make_scenario("identical_20_vs_20", obs_type="vector")
rng = jax.random.PRNGKey(3)
rng, key_reset = jax.random.split(rng, 2)
key_reset = jax.random.split(key_reset, batch_size)

obs, state = env.v_reset(key_reset)
ep_dones = jnp.zeros(batch_size, dtype=jnp.bool_)
t = 0
while not jnp.all(ep_dones):
    rng, key_action, key_step = jax.random.split(rng, 3)
    action = jax.random.uniform(
        key_action, shape=(batch_size, env.n_agents, 3), minval=-1, maxval=1
    )
    key_step = jax.random.split(key_step, batch_size)
    obs, state, rewards, dones, ep_dones = env.v_step(state, action, key_step)
    # obs is an uint8 array of shape [batch_size, n_agents, 84,84,3]
    # rewards is a float32 array of shape [batch_size, n_agents]
    # dones is a bool array of shape [batch_size, n_agents]
    # ep_done is a bool array of shape [batch_size]

    if t <= 1:
        print("obs.shape", obs.shape)
    t += 1
    # print(f"first agent's view {obs[0][0].reshape(15,9).transpose()}")

    # In case at least one episode is done, reset the state of the done episodes only
    if jnp.any(ep_dones):
        rng, key = jax.random.split(rng, 2)
        obs, states = env.reset_done_episodes(obs, state, ep_dones, key)
