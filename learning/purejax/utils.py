import os
import numpy as np
import cv2
import jax
import jax.numpy as jnp
from PIL import Image
from gigastep import stack_agents, GigastepEnv, GigastepViewer


def to_frame(
    dyn, state, step_obs, frame_size=4 * 84, obs1=True, obs2=False
):
    obs = dyn.get_global_observation(state)
    obs = np.array(obs, dtype=np.uint8)
    obs = cv2.resize(obs, (frame_size, frame_size), interpolation=cv2.INTER_NEAREST)

    image = obs
    if obs1:
        obs_1 = step_obs[0]
        obs_1 = np.array(obs_1, dtype=np.uint8)
        obs_1 = cv2.resize(
            obs_1, (frame_size, frame_size), interpolation=cv2.INTER_NEAREST
        )
        image = np.concatenate([image, obs_1], axis=1)
    if obs2:
        obs_2 = step_obs[1]
        obs_2 = np.array(obs_2, dtype=np.uint8)
        obs_2 = cv2.resize(
            obs_2, (frame_size, frame_size), interpolation=cv2.INTER_NEAREST
        )
        image = np.concatenate([image, obs_2], axis=1)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image


def get_ep_done(unwrapped_env, done):
    team_1 = 1 - unwrapped_env.teams
    team_1_done = jnp.sum(team_1[done]) == jnp.sum(team_1)
    team_2 = unwrapped_env.teams
    team_2_done = jnp.sum(team_2[done]) == jnp.sum(team_2)
    return (team_1_done or team_2_done).item()


def generate_gif(env_tuple, action_fn, filepath, seed=42,
                 max_frame_num=500, viewer_title="Evaluation"):
    rng = jax.random.PRNGKey(seed)
    env, unwrapped_env = env_tuple

    viewer = GigastepViewer(84 * 4, show_num_agents=0)
    viewer.set_title(viewer_title)

    key, rng = jax.random.split(rng, 2)
    frame_num = 0
    frame_list = []
    obs, state = env.reset(key)
    ep_done = False
    while not ep_done and frame_num < max_frame_num:
        rng, key = jax.random.split(rng, 2)
        action = action_fn(obs, key)
        rng, key = jax.random.split(rng, 2)
        obs, state, r, done, info = env.step(key, state, action)
        ep_done = get_ep_done(unwrapped_env, done)
        rgb_obs = env.get_global_observation(state.env_state)
        frame = viewer.draw(unwrapped_env, state.env_state, rgb_obs)
        frame_list.append(frame)
        frame_num += 1

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    imgs = [Image.fromarray(frame) for frame in frame_list]
    imgs[0].save(filepath, save_all=True, append_images=imgs[1:], duration=50, loop=0)
