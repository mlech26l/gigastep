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

from learning.purejax.ippo import ImageObsWrapper, FrameStackWrapper, Transition
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
        ### determine win rate
        team1 = self._env._env.teams  # env_state[0]["team"]
        team2 = 1 - self._env._env.teams  # env_state[0]["team"]
        team1_done = team1 * done
        team2_done = team2 * done
        team1_all_done = team1_done.sum() == team1.sum()
        team2_all_done = team2_done.sum() == team2.sum()
        ###
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        info["team1_all_done"] = team1_all_done
        info["team2_all_done"] = team2_all_done
        return obs, state, reward, done, info


def make_eval_fn(config):
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
            obs_shape=unwrapped_env.observation_space.shape,
        )
    elif config["NETWORK_TYPE"] == "lstm":
        # make_network = partial(ActorCriticLSTM, 128, unwrapped_env.action_space.n)
        raise NotImplementedError
    else:
        raise ValueError("Unrecognized network type " + config["NETWORK_TYPE"])

    def eval_fn(rng, net_params, env_state=None, obsv=None):
        network = make_network()

        if net_params is None:
            # INIT NETWORK
            rng, _rng = jax.random.split(rng)
            if config["FRAME_STACK_N"] > 1:
                init_x = jnp.zeros(
                    (unwrapped_env.n_agents,)
                    + unwrapped_env.observation_space.shape[:-1]
                    + (
                        unwrapped_env.observation_space.shape[-1]
                        * config["FRAME_STACK_N"],
                    )
                )
            else:
                init_x = jnp.zeros(
                    (unwrapped_env.n_agents,)
                    + unwrapped_env.observation_space.shape
                )
            net_params = network.init(_rng, init_x)

        if env_state is None:
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, None)

        def _env_step(runner_state, unused):
            rng, net_params, env_state, last_obs = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value = network.apply(net_params, last_obs)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            # env_state, obsv, reward, done, ep_done = env.v_step(env_state, action, rng_step)
            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(rng_step, env_state, action, None)
            transition = Transition(
                done, action, value, reward, log_prob, last_obs, info
            )
            runner_state = (rng, net_params, env_state, obsv)
            return runner_state, transition

        rng, _rng = jax.random.split(rng)
        runner_state = (_rng, net_params, env_state, obsv)
        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["EVAL_NUM_STEPS"]
        )
        metric = traj_batch.info
        return {"runner_state": runner_state, "metrics": metric}

    return eval_fn, (env, unwrapped_env), make_network


def load_ckpt(filepath):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt = orbax_checkpointer.restore(filepath)
    return ckpt


if __name__ == "__main__":
    ENV_NAME = ["identical_5_vs_5", "identical_20_vs_20", "identical_5_vs_1"][0]
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default=ENV_NAME)
    parser.add_argument("--ckpt1", type=str, default="")
    parser.add_argument("--ckpt2", type=str, default="")
    parser.add_argument(
        "--ckpt-mode", type=str, default="12", choices=["12", "11", "22", "21"]
    )
    parser.add_argument("--n-episodes", type=int, default=1000)
    parser.add_argument("--min-ep-len", type=int, default=10)
    args = parser.parse_args()
    ALL_TOTAL_TIMESTEPS = 2e7
    EVAL_EVERY = 2e6
    EVAL_N_EPS = 4
    resolution = 84
    config = {
        "ENV_CONFIG": {
            "resolution_x": resolution,
            "resolution_y": resolution,
            "obs_type": "vector",  # "rgb", # "vector",
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
        ###### FOR EVAL
        "EVAL_NUM_STEPS": 1e3,  # number of steps for each jitted eval (note that this is different from env max steps)
        "EVAL_TOTAL_NUM_STEPS": 1e5,  # total number of eval steps
    }

    rng = jax.random.PRNGKey(42)
    eval_fn, env_tuple, make_network = make_eval_fn(config)
    eval_fn_jit = jax.jit(eval_fn)

    # load network params
    ckpt1 = load_ckpt(args.ckpt1)
    ckpt2 = load_ckpt(args.ckpt2)

    network_params1 = ckpt1["model"]["params"]
    network_params2 = ckpt2["model"]["params"]
    network_params = {"params": {}}
    for k in network_params1["params"].keys():
        if args.ckpt_mode == "12":
            if "team2" in k:
                network_params["params"][k] = network_params2["params"][k]
            else:
                network_params["params"][k] = network_params1["params"][k]
        elif args.ckpt_mode == "11":
            if "team2" in k:
                network_params["params"][k] = network_params2["params"][
                    k.replace("team2", "team1")
                ]
            else:
                network_params["params"][k] = network_params1["params"][k]
        elif args.ckpt_mode == "22":
            if "team2" in k:
                network_params["params"][k] = network_params2["params"][k]
            else:
                network_params["params"][k] = network_params1["params"][
                    k.replace("team1", "team2")
                ]
        elif args.ckpt_mode == "21":
            if "team2" in k:
                network_params["params"][k] = network_params1["params"][k]
            else:
                network_params["params"][k] = network_params2["params"][k]
        else:
            raise ValueError(f"Unrecognized checkpoint mode {args.ckpt_mode}")

    win1_list = []  # wrt team1
    win2_list = []  # wrt team1
    ep_ret1_list = [] # epsiode return of team 1 averaged across agents in the same team
    ep_ret2_list = [] # epsiode return of team 1 averaged across agents in the same team
    pbar = tqdm(
        range(0, int(config["EVAL_TOTAL_NUM_STEPS"]), int(config["EVAL_NUM_STEPS"]))
    )
    pbar.set_description("[0/{}] -".format(config["EVAL_TOTAL_NUM_STEPS"]))
    out = {"runner_state": (rng, network_params, None, None)}
    for i in pbar:
        try:
            tic = time.time()
            out = eval_fn_jit(*out["runner_state"])
            toc = time.time()

            team1_all_done = np.asarray(out["metrics"]["team1_all_done"])
            team2_all_done = np.asarray(out["metrics"]["team2_all_done"])
            ep_done = team1_all_done | team2_all_done
            win_team1 = 1 - team1_all_done[ep_done]
            win_team2 = 1 - team2_all_done[ep_done]
            ep_len = np.asarray(out["metrics"]["returned_episode_lengths"]).max(-1)
            team1 = np.asarray(env_tuple[1].teams)
            team2 = 1 - team1
            ep_ret1 = np.asarray(out["metrics"]["returned_episode_returns"])[...,team1.astype(bool)].mean(-1) # average across agents
            ep_ret2 = np.asarray(out["metrics"]["returned_episode_returns"])[...,team2.astype(bool)].mean(-1) # average across agents
            ep_ret1 = ep_ret1[ep_done]
            ep_ret2 = ep_ret2[ep_done]
            if args.min_ep_len > 0:
                ep_len_done = ep_len[ep_done]
                win_team1 = win_team1[ep_len_done >= args.min_ep_len]
                win_team2 = win_team2[ep_len_done >= args.min_ep_len]
                ep_ret1 = ep_ret1[ep_len_done >= args.min_ep_len]
                ep_ret2 = ep_ret2[ep_len_done >= args.min_ep_len]
            win1_list.extend(win_team1.tolist())
            win2_list.extend(win_team2.tolist())
            ep_ret1_list.extend(ep_ret1.tolist())
            ep_ret2_list.extend(ep_ret2.tolist())
        except KeyboardInterrupt:
            break

        win1 = sum(win1_list)
        win2 = sum(win2_list)
        total = len(win1_list)
        win1_rate = win1 / total
        pbar.set_description(
            "[{}/{}] {}".format(i, config["EVAL_TOTAL_NUM_STEPS"], toc - tic)
            + f" win rate {win1_rate:.6f} ({win1}/{total})"
        )

        if total > args.n_episodes:
            break

    win1 = sum(win1_list[: args.n_episodes])
    win2 = sum(win2_list[: args.n_episodes])
    ep_ret1 = sum(ep_ret1_list[: args.n_episodes])
    ep_ret2 = sum(ep_ret2_list[: args.n_episodes])
    total = len(win1_list[: args.n_episodes])
    print(
        f"{args.env_name},{args.ckpt_mode},{win1},{win2},{ep_ret1},{ep_ret2},{total},{args.ckpt1},{args.ckpt2}"
    )