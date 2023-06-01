import jax
import jax.numpy as jnp
from typing import Optional
import chex

import gigastep

class GigastepFitness(object):
    def __init__(
        self,
        env_name: str = "identical_20_vs_20",
        num_env_steps: Optional[int] = 50, # 100,  # None,
        num_rollouts: int = 16,
        env_kwargs: dict = {},
        env_params: dict = {},
        test: bool = False,
        n_devices: Optional[int] = None,
        env_cfg: dict = {},
    ):
        self.env_name = env_name
        self.num_rollouts = num_rollouts
        self.test = test
        self.debug_reward = env_cfg["debug_reward"]

        # Define the RL environment & replace default parameters if desired
        self.env = gigastep.make_scenario(env_name, **env_cfg)

        self.num_env_steps = num_env_steps
        self.steps_per_member = self.num_env_steps * num_rollouts

        self.ego_team_size = jnp.sum(self.env.teams==0)
        self.ado_team_size = jnp.sum(self.env.teams==1)

        self.action_shape = self.env.action_space.shape
        self.input_shape = self.env.observation_space.shape
        if n_devices is None:
            self.n_devices = jax.local_device_count()
        else:
            self.n_devices = n_devices

        # Keep track of total steps executed in environment
        self.total_env_steps = 0

    def set_apply_fn(self, network_apply, carry_init=None):
        """Set the network forward function."""
        self.network = network_apply
        # Set rollout function based on model architecture
        if carry_init is not None:
            self.single_rollout = self.rollout_rnn
            self.carry_init = carry_init
        else:
            self.single_rollout = self.rollout_ffw
        self.rollout_repeats = jax.vmap(self.single_rollout, in_axes=(0, None, None))
        self.rollout_pop = jax.vmap(self.rollout_repeats, in_axes=(None, 0, 0))
        # pmap over popmembers if > 1 device is available - otherwise pmap
        if self.n_devices > 1:
            self.rollout_map = self.rollout_pmap
            print(
                f"GymFitness: {self.n_devices} devices detected. Please make"
                " sure that the ES population size divides evenly across the"
                " number of devices to pmap/parallelize over."
            )
        else:
            self.rollout_map = self.rollout_pop

    def rollout_pmap(
        self, rng_input: chex.PRNGKey, ego_policy_params: chex.ArrayTree, ado_policy_params: chex.ArrayTree
    ):
        """Parallelize rollout across devices. Split keys/reshape correctly."""
        keys_pmap = jnp.tile(rng_input, (self.n_devices, 1, 1))
        out_rollout_map = jax.pmap(self.rollout_pop)(
            keys_pmap, ego_policy_params, ado_policy_params
        )
        rew_re = out_rollout_map[0].reshape(-1, self.num_rollouts)
        steps_re = out_rollout_map[1].reshape(-1, self.num_rollouts)  # NOTE: weird shape, but will only be summed
        if self.test:
            images_global_re = out_rollout_map[2][0]  # NOTE: 1st device
            out_rollout = rew_re, steps_re, images_global_re
        else:
            out_rollout = rew_re, steps_re
        if self.debug_reward:
            reward_info = out_rollout_map[-1]
            reward_info = {k: v.mean() for k, v in reward_info.items()}
            out_rollout = *out_rollout, reward_info
            
        # rew_re = rew_dev.reshape(-1, self.num_rollouts)
        # steps_re = steps_dev.reshape(-1, self.num_rollouts)
        return out_rollout

    def rollout(self, rng_input: chex.PRNGKey, ego_policy_params: chex.ArrayTree, ado_policy_params: chex.ArrayTree):
        """Placeholder fn call for rolling out a population for multi-evals."""
        rng_pop = jax.random.split(rng_input, self.num_rollouts)
        out_rollout_map = jax.jit(self.rollout_map)(rng_pop, ego_policy_params, ado_policy_params)
        scores = out_rollout_map[0]
        masks = out_rollout_map[1]
        if self.test:
            images_global = out_rollout_map[2]
        # Update total step counter using only transitions before termination of all ego team agents
        self.total_env_steps += masks[..., :self.ego_team_size].sum(axis=-2).max(axis=-1).sum()
        if self.test:
            out_rollout = scores.mean(axis=-1), images_global
        else:
            out_rollout = scores.mean(axis=-1),
        if self.debug_reward:
            reward_info = out_rollout_map[-2]
            reward_info = {k: v.mean() for k, v in reward_info.items()}
            out_rollout = *out_rollout, reward_info
        act_info = out_rollout_map[-1]
        act_info = {k: v.mean(axis=(0, 1)) for k, v in act_info.items()}
        out_rollout = *out_rollout, act_info
        return out_rollout

    def rollout_ffw(
        self, rng_input: chex.PRNGKey, ego_policy_params: chex.ArrayTree, ado_policy_params: chex.ArrayTree
    ):
        """Rollout an episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        state, obs = self.env.reset(rng_reset)

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            obs, state, ego_policy_params, ado_policy_params, rng, cum_reward, valid_mask = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)

            obs_ego = obs[:self.ego_team_size]
            obs_ado = obs[self.ego_team_size:]

            action_ego, action_info = self.network(ego_policy_params, obs_ego, rng=rng_net)
            action_ado, _ = self.network(ado_policy_params, obs_ado, rng=rng_net)
            action = jnp.concatenate((action_ego, action_ado), axis=0)

            next_s, next_o, reward, dones, done = self.env.step(
                state, action, rng_step
            )

            next_o_global = self.env.get_global_observation(next_s)

            # reward = reward[..., :self.ego_team_size].mean(axis=-1)  # FIXME: mean over all agent reward [testing, not desired]
            new_cum_reward = cum_reward + (reward * valid_mask)[..., :self.ego_team_size].mean(axis=-1)
            new_valid_mask = valid_mask * (1 - dones)
            carry = [
                next_o.squeeze(),
                next_s,
                ego_policy_params,
                ado_policy_params,
                rng,
                new_cum_reward,
                new_valid_mask,
            ]
            if self.test:
                y = [new_valid_mask, next_o_global.squeeze()]
            else:
                y = [new_valid_mask]
            if self.debug_reward:
                reward_info = {k: v for k, v in next_s[0].items() if "reward" in k}
                y.append(reward_info)
            y.append(action_info)
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                ego_policy_params,
                ado_policy_params,
                rng_episode,
                jnp.array([0.0]),
                jnp.ones((obs.shape[0],)),
            ],
            (),
            self.num_env_steps,
        )
        # Return the sum of rewards accumulated by agent in episode rollout
        if self.test:
            ep_mask = scan_out[0]
            ep_obsv_global = scan_out[1]
            out_scan = (jnp.array(ep_mask), jnp.array(ep_obsv_global))
        else:
            ep_mask = scan_out[0]
            out_scan = (jnp.array(ep_mask),)
        if self.debug_reward:
            reward_info = {k: v[..., :self.ego_team_size].mean() for k, v in scan_out[-2].items()}
            out_scan = (*out_scan, reward_info)
        act_info = {k: v.mean(axis=(0, 1)) for k, v in scan_out[-1].items()}
        out_scan = (*out_scan, act_info)
        cum_return = carry_out[-2].squeeze()
        return cum_return, *out_scan

    def rollout_rnn(
        self, rng_input: chex.PRNGKey, ego_policy_params: chex.ArrayTree, ado_policy_params: chex.ArrayTree
    ):
        """Rollout a jitted episode with lax.scan."""
        # Reset the environment
        rng, rng_reset = jax.random.split(rng_input)
        state, obs = self.env.reset(rng_reset)

        hidden_ego = self.carry_init(batch_dims=(self.ego_team_size,))
        hidden_ado = self.carry_init(batch_dims=(self.ado_team_size,))

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            (
                obs,
                state,
                ego_policy_params,
                ado_policy_params,
                rng,
                hidden_ego,
                hidden_ado,
                cum_reward,
                valid_mask,
            ) = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)

            obs_ego = obs[:self.ego_team_size]
            obs_ado = obs[self.ego_team_size:]

            hidden_ego, action_ego, action_info = self.network(ego_policy_params, obs_ego, hidden_ego, rng_net)
            hidden_ado, action_ado, _ = self.network(ado_policy_params, obs_ado, hidden_ado, rng_net)
            action = jnp.concatenate((action_ego, action_ado), axis=0)

            next_s, next_o, reward, dones, done = self.env.step(
                state, action, rng_step
            )

            next_o_global = self.env.get_global_observation(next_s)

            new_cum_reward = cum_reward + (reward * valid_mask)[..., :self.ego_team_size].mean(axis=-1)
            new_valid_mask = valid_mask * (1 - dones)
            carry = [
                next_o.squeeze(),
                next_s,
                ego_policy_params,
                ado_policy_params,
                rng,
                hidden_ego,
                hidden_ado,
                new_cum_reward,
                new_valid_mask,
            ]
            if self.test:
                y = [new_valid_mask, next_o_global.squeeze()]
            else:
                y = [new_valid_mask]
            if self.debug_reward:
                reward_info = {k: v for k, v in next_s[0].items() if "reward" in k}
                y.append(reward_info)
            y.append(action_info)
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                ego_policy_params,
                ado_policy_params,
                rng,
                hidden_ego,
                hidden_ado,
                jnp.array([0.0]),
                jnp.ones((obs.shape[0],)),
            ],
            (),
            self.num_env_steps,
        )
        # Return masked sum of rewards accumulated by agent in episode
        if self.test:
            ep_mask = scan_out[0]
            ep_obsv_global = scan_out[1]
            out_scan = (jnp.array(ep_mask), jnp.array(ep_obsv_global))
        else:
            ep_mask = scan_out[0]
            out_scan = (jnp.array(ep_mask),)
        if self.debug_reward:
            reward_info = {k: v[..., :self.ego_team_size].mean() for k, v in scan_out[-2].items()}
            out_scan = (*out_scan, reward_info)
        act_info = {k: v.mean(axis=(0, 1)) for k, v in scan_out[-1].items()}
        out_scan = (*out_scan, act_info)
        cum_return = carry_out[-2].squeeze()
        return cum_return, *out_scan
