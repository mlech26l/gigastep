import os
import time
from tqdm import tqdm
from copy import deepcopy
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from typing import NamedTuple, Any
from flax.training.train_state import TrainState
import orbax.checkpoint
from flax.training import orbax_utils
from gigastep import make_scenario

import chex
from flax import struct
from functools import partial
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper
from typing import Optional, Tuple, Union

from utils import generate_gif, get_ep_done
from network import ActorCriticMLP, ActorCriticLSTM


class GigaStepWrapper(GymnaxWrapper):
    # NOTE: auto-reset; handle param args; change return order
    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        state, obs = self._env.reset(key)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        key, key_reset = jax.random.split(key)
        state_st, obs_st, reward, done, episode_done = self._env.step(
            state, action, key
        )
        obs_re, state_re = self.reset(key_reset, params)
        state = jax.tree_map(
            lambda x, y: jax.lax.select(episode_done, x, y), state_re, state_st
        )
        if isinstance(obs_re, tuple):
            obs = jax.tree_map(
                lambda x, y: jax.lax.select(episode_done, x, y), obs_re, obs_st
            )
        else:
            obs = jax.lax.select(episode_done, obs_re, obs_st)

        return obs, state, reward, done, {}  # replace last argument with empty info


# HACK
class GigaStepTupleObsWrapper(GigaStepWrapper):
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        key, key_reset = jax.random.split(key)
        state_st, obs_st, reward, done, episode_done = self._env.step(
            state, action, key
        )
        obs_re, state_re = self.reset(key_reset, params)
        state = jax.tree_map(
            lambda x, y: jax.lax.select(episode_done, x, y), state_re, state_st
        )
        obs = (
            jax.lax.select(episode_done, obs_re[0], obs_st[0]),
            jax.lax.select(episode_done, obs_re[1], obs_st[1]),
        )

        return obs, state, reward, done, {}  # replace last argument with empty info


# TODO: seems inefficient
class FrameStackWrapper(GymnaxWrapper):
    def __init__(self, env: environment.Environment, n_frames: int):
        super().__init__(env)
        self.n_frames = n_frames

    def _get_stacked_obs(self, obs, state):
        state[0]["stacked_obs"][self.pointer] = obs
        self.pointer = (self.pointer + 1) % self.n_frames
        return jnp.concatenate(state[0]["stacked_obs"], axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key)
        self.pointer = 0
        state[0]["stacked_obs"] = [obs] * self.n_frames
        obs = self._get_stacked_obs(obs, state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        stacked_obs = state[0]["stacked_obs"]
        obs, state, reward, done, _ = self._env.step(
            key,
            state,
            action,
        )
        state[0]["stacked_obs"] = stacked_obs
        obs = self._get_stacked_obs(obs, state)
        return obs, state, reward, done, {}  # replace last argument with empty info


class ImageObsWrapper(GymnaxWrapper):
    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key)
        obs = self._process_obs(obs)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action)
        obs = self._process_obs(obs)
        return obs, state, reward, done, info

    def _process_obs(self, obs):
        return obs.reshape(obs.shape[0], -1)


@struct.dataclass
class LogMultiAgentEnvState:
    env_state: environment.EnvState
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray
    returned_episode_returns: jnp.ndarray
    returned_episode_lengths: jnp.ndarray


class LogMultiAgentWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogMultiAgentEnvState(
            env_state,
            jnp.zeros(self._env.n_agents, dtype=jnp.float32),
            jnp.zeros(self._env.n_agents, dtype=jnp.int32),
            jnp.zeros(self._env.n_agents, dtype=jnp.float32),
            jnp.zeros(self._env.n_agents, dtype=jnp.int32),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogMultiAgentEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        return obs, state, reward, done, info


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_not_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    unwrapped_env = make_scenario(config["ENV_NAME"], **config["ENV_CONFIG"])
    env = GigaStepWrapper(unwrapped_env)
    if config["FRAME_STACK_N"] > 1:
        env = FrameStackWrapper(env, config["FRAME_STACK_N"])
    if config["ENV_CONFIG"]["obs_type"] == "rgb":
        env = ImageObsWrapper(env)
    env = LogMultiAgentWrapper(env)

    if config["NETWORK_TYPE"] == "mlp":
        make_network = partial(
            ActorCriticMLP,
            unwrapped_env.action_space.n,
            activation=config["ACTIVATION"],
            teams=unwrapped_env.teams,
            has_cnn=config["ENV_CONFIG"]["obs_type"] == "rgb" and config["USE_CNN"],
            obs_shape=unwrapped_env.observation_space[1].shape,
        )
    elif config["NETWORK_TYPE"] == "lstm":
        # make_network = partial(ActorCriticLSTM, 128, unwrapped_env.action_space.n)
        raise NotImplementedError
    else:
        raise ValueError("Unrecognized network type " + config["NETWORK_TYPE"])

    return (env, unwrapped_env), make_network


def get_filename(dir):
    file_id = 0
    while True:
        filename = os.path.join(dir, f"ep_{file_id:06d}.npz")
        if not os.path.exists(filename):
            return filename
        file_id += 1


def save_npz(filename, params):
    params = params["params"]
    team1 = {
        "w1": params["actor/team1/dense_0"]["kernel"],
        "b1": params["actor/team1/dense_0"]["bias"],
        "w2": params["actor/team1/dense_1"]["kernel"],
        "b2": params["actor/team1/dense_1"]["bias"],
        "w3": params["actor/team1/last"]["kernel"],
        "b3": params["actor/team1/last"]["bias"],
    }
    team2 = {
        "w1": params["actor/team2/dense_0"]["kernel"],
        "b1": params["actor/team2/dense_0"]["bias"],
        "w2": params["actor/team2/dense_1"]["kernel"],
        "b2": params["actor/team2/dense_1"]["bias"],
        "w3": params["actor/team2/last"]["kernel"],
        "b3": params["actor/team2/last"]["bias"],
    }
    np.savez(filename, team1=team1, team2=team2)


if __name__ == "__main__":
    ENV_NAME = ["identical_5_vs_5", "identical_20_vs_20", "identical_5_vs_1"][0]
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default=ENV_NAME)
    parser.add_argument("--ckpt", type=str, default="")
    args = parser.parse_args()
    ALL_TOTAL_TIMESTEPS = 2e7
    EVAL_EVERY = 2e6
    EVAL_N_EPS = 4
    resolution = 84
    config = {
        "ENV_CONFIG": {
            "resolution_x": resolution,
            "resolution_y": resolution,
            "obs_type": "rgb_vector",  # "rgb", # "vector",
            "discrete_actions": True,
            "reward_game_won": 100,
            "reward_defeat_one_opponent": 100,
            "reward_detection": 0,
            "reward_damage": 0,
            "reward_idle": 0,
            "reward_collision_agent": 0,
            "reward_collision_obstacle": 100,
            "cone_depth": 15.0,
            "cone_angle": jnp.pi * 1.99,
            "enable_waypoints": False,
            "use_stochastic_obs": False,
            "use_stochastic_comm": False,
            "max_agent_in_vec_obs": 100,
            "max_episode_length": 256,  # 1024,
        },
        "NETWORK_TYPE": ["mlp", "lstm"][0],
        "USE_CNN": False,
        "FRAME_STACK_N": 1,
        "LR": 4e-4,
        # each epoch has batch (experience replay buffer) size as num_envs * num_steps
        "NUM_ENVS": 64,
        "NUM_STEPS": 256,  # 1024, # 256,
        "TOTAL_TIMESTEPS": 1e5,  # 1e5, # for one train_jit only
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        # 4, # determine minibatch_size as buffer_size / num_minibatches; also num of minibatch size within an epoch
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.05,  # 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.1,  # 0.5,
        "ACTIVATION": ["relu", "tanh"][1],  # "tanh",
        "ENV_NAME": args.env_name,
        "ANNEAL_LR": False,  # True,
    }
    # python3 replay.py --env-name identical_5_vs_5 --ckpt logdir/all_with_new_rew_ver3/identical_5_vs_5/ckpt/20000000/
    rng = jax.random.PRNGKey(42)
    env_tuple, make_network = make_not_train(config)

    network = make_network()

    # load network params
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt = orbax_checkpointer.restore(args.ckpt)
    network_params = ckpt["model"]["params"]

    save_npz("pi1.npz", network_params)

    DETERMINISTIC_ACTION = False  # True

    def action_fn_base(network_params, obs, rng):
        rng, _rng = jax.random.split(rng)
        pi, value = network.apply(network_params, obs)
        if DETERMINISTIC_ACTION:
            action = pi.probs.argmax(1)
        else:
            action = pi.sample(seed=_rng)
        return action

    from gigastep import GigastepViewer

    viewer = GigastepViewer(84 * 10, show_global_state=False, show_num_agents=1)
    viewer.set_title("Replay")
    SAVE_GIF = False
    if SAVE_GIF:
        from PIL import Image

        frame_num = 0
        frame_list = []

    env, unwrapped_env = env_tuple

    while True:
        rng, _rng = jax.random.split(rng)
        (obs_rgb, obs_vec), state = env.reset(_rng)
        ego_done = False

        rgb_obs_buffer = []
        vec_obs_buffer = []
        action_buffer = []
        reward_buffer = []
        total_reward_0 = 0
        while not ego_done:
            action = action_fn_base(network_params, obs_vec, rng)
            action_ego = viewer.continuous_action
            action_ego = jnp.argmin(
                jnp.linalg.norm(action_ego[None, :] - unwrapped_env.action_lut, axis=1)
            )
            action_buffer.append(action_ego)
            rgb_obs_buffer.append(obs_rgb[0])
            vec_obs_buffer.append(obs_vec[0])

            is_ego = jnp.arange(action.shape[0]) == 0
            # print(
            #     f"action_ego: {action_ego} ({unwrapped_env.action_lut[action_ego]} from {viewer.continuous_action})"
            # )
            action = jnp.where(is_ego, action_ego, action)
            rng, _rng = jax.random.split(rng)
            obs, state, r, done, info = env.step(_rng, state, action)
            obs_rgb, obs_vec = obs
            # ep_done = get_ep_done(unwrapped_env, done)
            ego_done = done[0]

            total_reward_0 += r[0]
            reward_buffer.append(r[0])

            rgb_obs = env.get_global_observation(state.env_state)
            frame = viewer.draw(unwrapped_env, state.env_state, obs_rgb)
            if SAVE_GIF:
                frame_list.append(frame)
                frame_num += 1
            if viewer.should_quit:
                import sys

                sys.exit(0)
        print(
            f"Episode done, total reward: {total_reward_0:0.2f} ({len(action_buffer)} ep len)"
        )
        os.makedirs("human_data", exist_ok=True)
        filename = get_filename("human_data")
        rgb_obs_buffer = np.stack(rgb_obs_buffer, axis=0)
        vec_obs_buffer = np.stack(vec_obs_buffer, axis=0)
        action_buffer = np.array(action_buffer)
        reward_buffer = np.array(reward_buffer)
        np.savez(
            filename,
            rgb_obs=rgb_obs_buffer,
            vec_obs=vec_obs_buffer,
            rewards=reward_buffer,
            action=action_buffer,
        )
        with open("human_data/logs.txt", "a") as f:
            f.write(f"{filename}, {total_reward_0:0.2f}, {action_buffer.shape[0]}\n")
        print(f"Saved to {filename}")

        if SAVE_GIF:
            filepath = "./logdir/test.gif"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            imgs = [Image.fromarray(frame) for frame in frame_list]
            imgs[0].save(
                filepath, save_all=True, append_images=imgs[1:], duration=50, loop=0
            )