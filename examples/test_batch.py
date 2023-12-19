from gigastep import make_scenario, GigastepViewer
import jax
import jax.numpy as jnp
import cv2
import numpy as np
import torch
import os
import sys
batch_size = 1
viewer = GigastepViewer(frame_size=200, show_num_agents=1)

env = make_scenario("identical_20_vs_20",obs_type="vector")
rng = jax.random.PRNGKey(3)
rng, key_reset = jax.random.split(rng, 2)
key_reset = jax.random.split(key_reset, batch_size)

obs, state = env.v_reset(key_reset)
ep_dones = jnp.zeros(batch_size, dtype=jnp.bool_)
load_policy = False
if load_policy:
    load_dir = "trained_models/ppogigastepPro20230416-122744.414250.pt"
    actor_critic, obs_rms = torch.load(os.path.join(load_dir),map_location='cpu')
    recurrent_hidden_states = torch.zeros(1,actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)

def get_action_jax(actor_critic, obs,recurrent_hidden_states):
    obs = torch.tensor(np.asarray(obs))
    obs = torch.moveaxis(obs, -1,2)
    obs = obs.view(int(obs.shape[0]*2),int(obs.shape[1]/2),*obs.shape[2:])
    masks = torch.zeros(1, 1)
    value, action, _, recurrent_hidden_states = actor_critic.act(
        obs, recurrent_hidden_states, masks, deterministic=True)

    action_view = action.view(int(action.shape[0] / 2), int(action.shape[1] * 2), *action.shape[2:])
    action_jax = jnp.array(action_view.detach().cpu().numpy().astype(np.uint32))
    return action_jax, recurrent_hidden_states

while True:
    print(f"all {jnp.all(ep_dones)}")
    rng, key_action,key_step = jax.random.split(rng, 3)
    if load_policy:
        action, recurrent_hidden_states = get_action_jax(actor_critic, obs, recurrent_hidden_states)
    else:
        action = jax.random.uniform(key_action, shape=(batch_size, env.n_agents, 3), minval=-1, maxval=1)
        action = jax.numpy.zeros((batch_size, env.n_agents, 3))
    key_step = jax.random.split(key_step, batch_size)
    obs, state, rewards, dones, ep_dones = env.v_step(state, action, key_step)

    if jnp.any(ep_dones):
        rng, key = jax.random.split(rng, 2)
        obs, state = env.reset_done_episodes(obs, state, ep_dones, key)

    if viewer.should_quit:
        sys.exit(1)
    # print(state[0]["team"])
    print(rewards)
    # obs is an uint8 array of shape [batch_size, n_agents, 84,84,3]
    # rewards is a float32 array of shape [batch_size, n_agents]
    # dones is a bool array of shape [batch_size, n_agents]
    # ep_done is a bool array of shape [batch_size]