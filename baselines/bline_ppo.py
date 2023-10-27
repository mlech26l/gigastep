from typing import Callable
import jax
import jax.numpy as jnp
from functools import partial
from flax import struct
from tqdm.auto import tqdm
import numpy as np


class AdversaryPolicy(struct.PyTreeNode):
    init_fn: Callable = struct.field(pytree_node=False)
    sample_fn: Callable = struct.field(pytree_node=False)
    params = None

    def init_state(self, batch_size, key):
        return self.init_fn(batch_size, key)

    def sample_actions(self, state, obs, key):
        return self.sample_fn(state, obs, key)

        # batch_size = obs.shape[0]
        # return (
        #     jax.random.randint(
        #         key,
        #         (batch_size, self.env.n_agents),
        #         0,
        #         self.n_actions,
        #     ),
        #     state,
        # )


def int_to_str(x):
    if x >= 1e9:
        return f"{x/1e9:.2g}B"
    if x >= 1e7:
        return f"{x/1e6:.0f}M"
    if x >= 1e6:
        return f"{x/1e6:.1f}M"
    if x >= 1e4:
        return f"{x/1e3:.0f}k"
    if x >= 1e3:
        return f"{x/1e3:.1f}k"
    return f"{x}"


def compute_gae(rollout_buffer, value_state, gamma=0.99, gae_lambda=0.95):
    def step_gae(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        obs, reward, alive = transition

        value = value_state.apply_fn(value_state.params, obs)[:, :, 0]
        delta = reward + gamma * next_value * alive - value
        gae = delta + gamma * gae_lambda * value * gae
        return (gae, value), (gae, value)

    last_value = value_state.apply_fn(value_state.params, rollout_buffer.obs[-1])[
        :, :, 0
    ]
    _, (advantages, values) = jax.lax.scan(
        step_gae,
        (jnp.zeros_like(last_value), last_value),
        [rollout_buffer.obs[:-1], rollout_buffer.rewards, rollout_buffer.alive],
        reverse=True,
    )
    return advantages, values, advantages + values


def flatten_time_batch_agent_dim(x):
    return jnp.reshape(x, (-1,) + x.shape[2:])


def shuffle_buffer(buffer, key):
    return jax.tree_util.tree_map(lambda x: jax.random.permutation(key, x), buffer)


def concat_buffers(buffer1, buffer2):
    if buffer1 is None:
        return buffer2
    return jax.tree_util.tree_multimap(
        lambda x, y: jnp.concatenate([x, y], axis=0), buffer1, buffer2
    )


def minibatchify(buffer, batch_size):
    trim_size = buffer.obs.shape[0] - buffer.obs.shape[0] % batch_size
    buffer = jax.tree_util.tree_map(lambda x: x[:trim_size], buffer)
    buffer = jax.tree_util.tree_map(
        lambda x: x.reshape((batch_size, -1) + x.shape[1:]), buffer
    )
    return buffer


class TrainBuffer(struct.PyTreeNode):
    # Store processed rollout data, ready for training
    obs: jnp.ndarray = struct.field(pytree_node=True)
    next_obs: jnp.ndarray = struct.field(pytree_node=True)
    actions: jnp.ndarray = struct.field(pytree_node=True)
    logprobs: jnp.ndarray = struct.field(pytree_node=True)
    value_targets: jnp.ndarray = struct.field(pytree_node=True)
    advantages: jnp.ndarray = struct.field(pytree_node=True)
    values: jnp.ndarray = struct.field(pytree_node=True)


class RolloutBuffer(struct.PyTreeNode):
    # Store raw ollout data (not yet processed for training)
    obs: jnp.ndarray = struct.field(pytree_node=True)
    actions: jnp.ndarray = struct.field(pytree_node=True)
    logprobs: jnp.ndarray = struct.field(pytree_node=True)
    rewards: jnp.ndarray = struct.field(pytree_node=True)
    alive: jnp.ndarray = struct.field(pytree_node=True)
    ep_dones: jnp.ndarray = struct.field(pytree_node=True)

    def average_episode_length(self):
        return self.ep_dones.sum(axis=0).mean()

    def average_episode_reward(self):
        return self.rewards.sum(axis=0).mean()

    def make_train_buffer(self, value_state, gamma, gae_lambda):
        advantages, values, value_targets = compute_gae(
            self, value_state, gamma, gae_lambda
        )
        obs = self.obs[:-1]
        next_obs = self.obs[1:]

        obs = flatten_time_batch_agent_dim(obs)
        next_obs = flatten_time_batch_agent_dim(next_obs)
        actions = flatten_time_batch_agent_dim(self.actions)
        logprobs = flatten_time_batch_agent_dim(self.logprobs)
        value_targets = flatten_time_batch_agent_dim(value_targets)
        advantages = flatten_time_batch_agent_dim(advantages)
        values = flatten_time_batch_agent_dim(values)
        alive = flatten_time_batch_agent_dim(self.alive)

        # Value network needs (obs, cum_rewards)
        obs = obs[alive]
        next_obs = next_obs[alive]
        actions = actions[alive]
        logprobs = logprobs[alive]
        value_targets = value_targets[alive]
        values = values[alive]
        advantages = advantages[alive]

        return TrainBuffer(
            obs=obs,
            next_obs=next_obs,
            actions=actions,
            logprobs=logprobs,
            value_targets=value_targets,
            values=values,
            advantages=advantages,
        )


def make_random_adversary(n_agents, n_actions):
    def init_fn(batch_size, key):
        return jnp.empty(())

    def sample_fn(state, obs, key):
        batch_size = obs.shape[0]
        return (
            jax.random.randint(
                key,
                (batch_size, n_agents),
                0,
                n_actions,
            ),
            state,
        )

    return AdversaryPolicy(init_fn, sample_fn)


def make_circling_adversary(n_agents, all_same_diretion=False):
    def init_fn(batch_size, key):
        left = 10
        right = 16
        direction = jax.random.randint(key, (batch_size, n_agents), 0, 2)
        direction = direction * (right - left) + left
        return direction

    def init_fn_same_dir(batch_size, key):
        left = 10
        right = 16
        direction = jax.random.randint(key, (batch_size, 1), 0, 2)
        direction = direction * (right - left) + left
        direction = jnp.repeat(direction, n_agents, axis=1)
        return direction, direction

    def sample_fn(state, obs, key):
        return state

    return AdversaryPolicy(
        init_fn_same_dir if all_same_diretion else init_fn, sample_fn
    )


def sample_from_policy(train_state, obs, key):
    pi = train_state.apply_fn(train_state.params, obs)
    action = pi.sample(seed=key)
    log_prob = pi.log_prob(action)
    return action, log_prob


class RolloutManager:
    def __init__(self, env, batch_size):
        self.env = env
        self.batch_size = batch_size  # batch_size must be static

    @partial(jax.jit, static_argnums=(0,))
    def generate(self, train_state, adv_policy, rng):
        rng, key, key_adv = jax.random.split(rng, 3)
        key = jax.random.split(key, self.batch_size)
        obs, state = self.env.v_reset(key)
        adv_state = adv_policy.init_state(self.batch_size, key_adv)

        def polcy_env_step(carry_in, _):
            obs, state, train_state, adv_state, rng = carry_in
            obs_ego, obs_adv = jnp.split(obs, [self.env.n_agents_team1], axis=1)
            rng, key_action, key_adv, key_step = jax.random.split(rng, 4)

            action_pi, logp = sample_from_policy(train_state, obs_ego, key_action)
            adv_action, adv_state = adv_policy.sample_actions(
                adv_state, obs_adv, key_adv
            )

            action_fused = jnp.concatenate([action_pi, adv_action], axis=-1)

            key_step = jax.random.split(key_step, self.batch_size)
            next_obs, state, rewards, alive, ep_dones = self.env.v_step(
                state, action_fused, key_step
            )
            carry, y = [next_obs, state, train_state, adv_state, rng], [
                [obs, action_pi, logp, rewards, alive, ep_dones]
            ]
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            polcy_env_step,
            [obs, state, train_state, adv_state, rng],
            None,
            length=100,
        )
        last_obs, last_state, train_state, adv_state, rng = carry_out
        obs, action_pi, logp, rewards, alive, ep_dones = scan_out[0]
        obs = jnp.concatenate([obs, last_obs[None]], axis=0)
        obs = obs[:, :, : self.env.n_agents_team1]
        rewards = rewards[:, :, : self.env.n_agents_team1]
        alive = alive[:, :, : self.env.n_agents_team1]

        return RolloutBuffer(
            obs=obs,
            actions=action_pi,
            logprobs=logp,
            rewards=rewards,
            alive=alive,
            ep_dones=ep_dones,
        )


def train_iter(buffer, policy_state, value_state, key, config):
    buffer = shuffle_buffer(buffer, key)
    buffer = minibatchify(buffer, config["train_batch_size"])

    def value_train_step(train_state, input_batch):
        obs, targets, values = input_batch

        def value_loss_fn(params):
            pred_values = train_state.apply_fn(params, obs)[:, 0]

            value_pred_clipped = values + (pred_values - values).clip(
                -config["value_clip_eps"], config["value_clip_eps"]
            )
            assert value_pred_clipped.shape == pred_values.shape
            assert pred_values.shape == targets.shape
            value_losses = jnp.square(pred_values - targets)
            value_losses_clipped = jnp.square(value_pred_clipped - targets)
            loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

            return loss

        grad_fn = jax.value_and_grad(value_loss_fn, has_aux=False)
        total_loss, grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, total_loss

    value_state, value_loss = jax.lax.scan(
        value_train_step, value_state, (buffer.obs, buffer.value_targets, buffer.values)
    )

    def policy_train_step(train_state, input_batch):
        obs, advantage, action, old_logprob = input_batch
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        def policy_loss_fn(params):
            pi = train_state.apply_fn(params, obs)
            logprob = pi.log_prob(action)

            logprob_ratio = logprob - old_logprob
            ratio = jnp.exp(logprob_ratio)
            loss_actor1 = ratio * advantage
            loss_actor2 = (
                jnp.clip(
                    ratio,
                    1.0 - config["clip_eps"],
                    1.0 + config["clip_eps"],
                )
                * advantage
            )
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
            loss_actor = loss_actor.mean()
            entropy = pi.entropy().mean()

            loss = loss_actor - config["entropy_coef"] * entropy
            return loss

        grad_fn = jax.value_and_grad(policy_loss_fn, has_aux=False)
        total_loss, grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, total_loss

    policy_state, policy_loss = jax.lax.scan(
        policy_train_step,
        policy_state,
        (buffer.obs, buffer.advantages, buffer.actions, buffer.logprobs),
    )
    loss_dict = {"policy_loss": policy_loss, "value_loss": value_loss}
    loss_dict = jax.tree_map(lambda x: x.mean(), loss_dict)
    return (
        policy_state,
        value_state,
        loss_dict,
    )


class Runner:
    def __init__(self, env, policy_state, value_state, config, rng=None):
        self.env = env
        self.policy_state = policy_state
        self.value_state = value_state
        self.config = config
        if rng is None:
            rng = jax.random.PRNGKey(0)
        self.rng = rng
        self.rollout_manager = RolloutManager(env, config["rollout_batch_size"])
        self.adversary_pool = [
            make_random_adversary(env.n_agents_team2, env.n_actions),
            # make_circling_adversary(env.n_agents_team2, all_same_diretion=True),
            # make_circling_adversary(env.n_agents_team2, all_same_diretion=False),
        ]
        self.total_samples_trained = 0
        self.pbar = None

    def rng_key(self, n=1):
        keys = jax.random.split(self.rng, n + 1)
        self.rng = keys[-1]
        if n == 1:
            return keys[0]
        return keys[:-1]

    def run(self):
        for i in range(self.config["ppo_iters"]):
            self.run_one_iter()

    def run_one_iter(self):
        if self.pbar is None:
            self.pbar = tqdm(total=self.config["ppo_iters"])
        train_buffer = None
        inner_pbar = tqdm(
            total=len(self.adversary_pool),
            desc=f"Generating rollouts with {len(self.adversary_pool)} adversaries",
            leave=True,
        )
        avg_ep_len, avg_ep_reward = [], []
        for i, adversary in enumerate(self.adversary_pool):
            buffer = self.rollout_manager.generate(
                self.policy_state, adversary, self.rng_key()
            )
            avg_ep_len.append(buffer.average_episode_length())
            avg_ep_reward.append(buffer.average_episode_reward())

            # Let's convert the raw rollout buffer into a train buffer
            buffer = buffer.make_train_buffer(
                self.value_state,
                gamma=self.config["gamma"],
                gae_lambda=self.config["gae_lambda"],
            )
            train_buffer = concat_buffers(train_buffer, buffer)
            inner_pbar.set_description(
                f"Generating rollouts, {int_to_str(len(train_buffer.obs))} samples collected (avg_ep_len={np.mean(avg_ep_len).item():.2f}, avg_ep_reward={np.mean(avg_ep_reward).item():.2f})"
            )
            inner_pbar.update(1)
        inner_pbar.close()
        inner_pbar = tqdm(
            total=self.config["epochs_per_iter"],
            desc=f"Training ({int_to_str(self.total_samples_trained)} samples so far)",
            leave=True,
        )
        for i in range(self.config["epochs_per_iter"]):
            self.policy_state, self.value_state, loss_dict = train_iter(
                train_buffer,
                self.policy_state,
                self.value_state,
                self.rng_key(),
                self.config,
            )
            inner_pbar.set_description(
                f"Training ({int_to_str(self.total_samples_trained)} samples so far), policy_loss={loss_dict['policy_loss'].item():.2f}, value_loss={loss_dict['value_loss'].item():.2f}"
            )
            inner_pbar.update(1)
        inner_pbar.close()
        self.total_samples_trained += len(train_buffer.obs)
        self.pbar.set_description(
            f"Total samples trained: {int_to_str(self.total_samples_trained)}"
        )
        self.pbar.update(1)
