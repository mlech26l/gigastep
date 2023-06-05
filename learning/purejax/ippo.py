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

from utils import generate_gif
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


def make_train(config):
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

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng, train_state=None, env_state=None, obsv=None, net_state=None):
        network = make_network()

        if train_state is None:
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
                if (config["ENV_CONFIG"]["obs_type"] == "rgb") and not config[
                    "USE_CNN"
                ]:  # HACK
                    init_x = jnp.zeros(
                        (unwrapped_env.n_agents,)
                        + (np.prod(unwrapped_env.observation_space.shape),)
                    )
                else:
                    init_x = jnp.zeros(
                        (unwrapped_env.n_agents,)
                        + unwrapped_env.observation_space.shape
                    )
            network_params = network.init(_rng, init_x)
            if config["ANNEAL_LR"]:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(config["LR"], eps=1e-5),
                )
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )

            if config["NETWORK_TYPE"] == "lstm":
                # TODO: tbu
                net_state = network.initial_state(None)
            else:
                net_state = jnp.zeros((config["NUM_ENVS"],))

            # INIT ENV
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            # env_state, obsv = env.v_reset(reset_rng)
            obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, None)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, net_state, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                if config["NETWORK_TYPE"] == "lstm":
                    pi, value = network.apply(train_state.params, last_obs)  # DEBUG
                else:  # mlp
                    pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                # env_state, obsv, reward, done, ep_done = env.v_step(env_state, action, rng_step)
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, None)
                if config["NETWORK_TYPE"] == "lstm":
                    import ipdb

                    ipdb.set_trace()  # TODO: reset rnn state
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, net_state, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, net_state, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        log_ratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(log_ratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # approx_kl = jnp.mean(ratio - 1.0 - log_ratio)

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            runner_state = (train_state, env_state, last_obs, net_state, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, net_state, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train, (env, unwrapped_env), make_network


if __name__ == "__main__":
    ENV_NAME = ["identical_5_vs_5", "identical_20_vs_20", "identical_5_vs_1"][0]
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default=ENV_NAME)
    args = parser.parse_args()
    BASE_DIR = f"./logdir/all_with_new_rew_ver3/{args.env_name}"  # "./logdir/exp_42"
    ALL_TOTAL_TIMESTEPS = 10e7
    EVAL_EVERY = 1e7
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
        "NUM_MINIBATCHES": 32,  # 4, # determine minibatch_size as buffer_size / num_minibatches; also num of minibatch size within an epoch
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
    rng = jax.random.PRNGKey(32)
    train, env_tuple, make_network = make_train(config)
    train_jit = jax.jit(train)

    if EVAL_EVERY > 0:
        network = make_network()

        def action_fn_base(network_params, obs, rng):
            rng, _rng = jax.random.split(rng)
            pi, value = network.apply(network_params, obs)
            action = pi.sample(seed=_rng)
            return action

    ep_ret_list = []
    pbar = tqdm(range(0, int(ALL_TOTAL_TIMESTEPS), int(config["TOTAL_TIMESTEPS"])))
    pbar.set_description(f"[0/{ALL_TOTAL_TIMESTEPS}] -")
    for i in pbar:
        try:
            tic = time.time()
            if i == 0:
                out = train_jit(rng)
            else:
                train_state, env_state, obsv, net_state, rng = out["runner_state"]
                out = train_jit(rng, train_state, env_state, obsv, net_state)
            toc = time.time()

            ep_ret_i = (
                out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1).tolist()
            )
            ep_ret_list.extend(ep_ret_i)

            pbar.set_description(f"[{i}/{ALL_TOTAL_TIMESTEPS}] {toc-tic}")

            current_ts = i + int(config["TOTAL_TIMESTEPS"])
            if (EVAL_EVERY > 0) and ((i == 0) or (current_ts % EVAL_EVERY == 0)):
                if i != 0:
                    ckpt = {"model": train_state, "config": config}
                    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                    save_args = orbax_utils.save_args_from_target(ckpt)
                    filepath = os.path.join(BASE_DIR, "ckpt", f"{current_ts:07d}")
                    orbax_checkpointer.save(filepath, ckpt, save_args=save_args)

                action_fn = partial(action_fn_base, out["runner_state"][0].params)
                for ii in range(EVAL_N_EPS):
                    filepath = os.path.join(
                        BASE_DIR, "video", f"{current_ts:07d}_{ii:02d}.gif"
                    )
                    generate_gif(
                        env_tuple,
                        action_fn,
                        filepath,
                        max_frame_num=config["ENV_CONFIG"]["max_episode_length"],
                        seed=42 + ii,
                    )
        except KeyboardInterrupt:
            break

    plt.plot(ep_ret_list)
    plt.xlabel("Timesteps")
    plt.ylabel("Return")
    plt.savefig(os.path.join(BASE_DIR, "return.png"))

    for eval_i in range(8):
        action_fn = partial(action_fn_base, out["runner_state"][0].params)
        filepath = os.path.join(BASE_DIR, "video", f"eval_{eval_i:02d}.gif")
        generate_gif(
            env_tuple,
            action_fn,
            filepath,
            seed=eval_i,
            max_frame_num=config["ENV_CONFIG"]["max_episode_length"],
        )