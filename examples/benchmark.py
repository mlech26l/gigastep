import argparse
import time

import jax
import jax.numpy as jnp
from flax.training import train_state

from gigastep import GigastepEnv
import time
import time
import optax
from flax import linen as nn


def create_train_state(model, rng, in_dim):
    """Creates initial `TrainState`."""
    params = model.init(rng, jnp.ones(in_dim))
    tx = optax.adam(3e-4)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def run_vmapped_no_scan(env, params, batch_size, n_steps, repeats=1):
    rng = jax.random.PRNGKey(1)
    for _ in range(repeats):
        rng, key = jax.random.split(rng, 2)
        key = jax.random.split(key, batch_size)
        states, obs = env.v_reset(key)
        for i in range(n_steps):
            rng, key = jax.random.split(rng, 2)
            if params is None:
                actions = jnp.zeros(
                    (batch_size, env.n_agents, env.action_space.shape[0])
                )
            else:
                actions = params.apply_fn(params.params, obs)
            key = jax.random.split(key, batch_size)
            states, obs, r, a, d = env.v_step(states, actions, key)
    obs.block_until_ready()


def run_vmapped_scan(env, params, batch_size, n_steps, repeats=1):
    rng = jax.random.PRNGKey(1)

    def policy_step(state_carry, tmp):
        states, obs, rng = state_carry
        if params is None:
            actions = jnp.zeros((batch_size, env.n_agents, env.action_space.shape[0]))
        else:
            actions = params.apply_fn(params.params, obs)
        rng, key = jax.random.split(rng, 2)
        key = jax.random.split(key, batch_size)
        states, obs, r, a, d = env.v_step(states, actions, key)
        carry = [states, obs, rng]
        return carry, []

    # Scan over episode step loop
    for _ in range(repeats):
        rng, key = jax.random.split(rng, 2)
        key = jax.random.split(key, batch_size)
        states, obs = env.v_reset(key)
        _, scan_out = jax.lax.scan(policy_step, [states, obs, rng], [], length=n_steps)
    scan_out[1].block_until_ready()


def run_single_no_scan(env, params, n_steps, repeats=1):
    rng = jax.random.PRNGKey(1)
    for _ in range(repeats):
        rng, key = jax.random.split(rng, 2)
        states, obs = env.reset(key)
        for i in range(n_steps):
            rng, key = jax.random.split(rng, 2)
            if params is None:
                actions = jnp.zeros((env.n_agents, env.action_space.shape[0]))
            else:
                actions = params.apply_fn(params.params, obs)
            states, obs, r, a, d = env.step(states, actions, key)
    obs.block_until_ready()


def run_single_scan(env, params, n_steps, repeats=1):
    rng = jax.random.PRNGKey(1)

    def policy_step(state_carry, tmp):
        states, obs, rng = state_carry
        if params is None:
            actions = jnp.zeros((env.n_agents, env.action_space.shape[0]))
        else:
            actions = params.apply_fn(params.params, obs)
        rng, key = jax.random.split(rng, 2)
        states, obs, r, a, d = env.step(states, actions, key)
        carry = [states, obs, rng]
        return carry, []

    # Scan over episode step loop
    for _ in range(repeats):
        rng, key = jax.random.split(rng, 2)
        states, obs = env.reset(key)
        _, scan_out = jax.lax.scan(policy_step, [states, obs, rng], [], length=n_steps)

    scan_out[1].block_until_ready()


def print_step_per_second(step_per_second):
    postfix = ""
    orig_step_per_second = step_per_second
    if step_per_second > 1e9:
        postfix = "G"
        step_per_second = step_per_second / 1e9
    elif step_per_second > 1e6:
        postfix = "M"
        step_per_second = step_per_second / 1e6
    elif step_per_second > 1e3:
        postfix = "k"
        step_per_second = step_per_second / 1e3
    print(f"{step_per_second:0.1f}{postfix} steps per second", end=" ")
    sec_to_100M = 100_000_000 / orig_step_per_second
    if sec_to_100M >= 60 * 60:
        hours = sec_to_100M / 60 / 60
        mins = (hours - int(hours)) * 60
        print(f" (time to 100M steps: {int(hours)} hours {mins:0.0f} minutes)", end="")
    elif sec_to_100M >= 60:
        mins = sec_to_100M / 60
        secs = (mins - int(mins)) * 60
        print(f" (time to 100M steps: {int(mins)} minutes {secs:0.0f} seconds)", end="")
    else:
        print(f" (time to 100M steps: {sec_to_100M:0.1f} seconds)", end="")
    print()


class GigastepTimer:
    def __init__(self, name, n_steps, batch_size, n_agents, obs_type):
        # example file or database connection
        self.name = name
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_agents = n_agents
        self.obs_type = obs_type

        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = time.time() - self.start_time
        num_agent_steps = self.n_steps * self.batch_size * self.n_agents
        step_per_second = num_agent_steps / elapsed
        print(f"{self.name}:", end=" ")
        print_step_per_second(step_per_second)

        return True


class ConvPolicy(nn.Module):
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), name="conv1")(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=64,
            kernel_size=(5, 5),
            strides=(2, 2),
            name="conv2",
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=128,
            kernel_size=(5, 5),
            strides=(2, 2),
            name="conv3",
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=128,
            kernel_size=(5, 5),
            strides=(2, 2),
            name="conv4",
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=128, name="hidden")(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_outputs, name="logits")(x)
        return x


class MLPPolicy(nn.Module):
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128, name="hidden")(x)
        x = nn.tanh(x)
        x = nn.Dense(
            features=self.num_outputs,
            name="logits",
        )(x)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4096, type=int)  # in minutes
    parser.add_argument("--repeats", default=5, type=int)  # in minutes
    parser.add_argument("--n_agents", default=20, type=int)  # in minutes
    parser.add_argument("--n_steps", default=2000, type=int)  # in minutes
    parser.add_argument("--hidden", default=128, type=int)  # in minutes
    parser.add_argument("--obs_type", default="vector")  # in minutes
    args = parser.parse_args()
    env = GigastepEnv(n_agents=args.n_agents, obs_type=args.obs_type)

    print(
        f"Running with {args.n_agents} agents, {args.n_steps} steps, {args.repeats} repeats, {args.batch_size} batch size, {args.obs_type} obs type"
    )
    if args.obs_type == "vector":
        policy = MLPPolicy(num_outputs=env.action_space.shape[0])
        input_shape = (1, env.observation_space.shape[0])
    else:
        policy = ConvPolicy(num_outputs=env.action_space.shape[0])
        input_shape = (1, 84, 84, 4)
    nn_state = create_train_state(policy, jax.random.PRNGKey(0), input_shape)
    nn_state = None
    with GigastepTimer(
        "single scan", args.n_steps * args.repeats, 1, args.n_agents, args.obs_type
    ):
        run_single_scan(env, nn_state, args.n_steps, args.repeats)
    with GigastepTimer(
        "single no scan", args.n_steps * args.repeats, 1, args.n_agents, args.obs_type
    ):
        run_single_no_scan(env, nn_state, args.n_steps, args.repeats)
    # with GigastepTimer(
    #     "vmapped scan",
    #     args.n_steps,
    #     args.batch_size * args.repeats,
    #     args.n_agents,
    #     args.obs_type,
    # ):
    #     run_vmapped_scan(
    #         env, nn_state, args.n_steps * args.repeats, args.batch_size, args.repeats
    #     )
    with GigastepTimer(
        "vmapped no scan",
        args.n_steps,
        args.batch_size * args.repeats,
        args.n_agents,
        args.obs_type,
    ):
        run_vmapped_no_scan(
            env, nn_state, args.n_steps * args.repeats, args.batch_size, args.repeats
        )


if __name__ == "__main__":
    main()