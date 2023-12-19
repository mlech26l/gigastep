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
    return train_state.TrainState.create(
        apply_fn=jax.jit(model.apply), params=params, tx=tx
    )


def run_vmapped_no_scan(env, params, batch_size, n_steps, repeats=1):
    rng = jax.random.PRNGKey(1)
    for _ in range(repeats):
        rng, key = jax.random.split(rng, 2)
        key = jax.random.split(key, batch_size)
        obs, states = env.v_reset(key)
        for i in range(n_steps):
            rng, key = jax.random.split(rng, 2)
            if params is None:
                actions = jnp.zeros(
                    (batch_size, env.n_agents, env.action_space.shape[0])
                )
            else:
                actions = params.apply_fn(params.params, obs)
            key = jax.random.split(key, batch_size)
            obs, states, r, a, d = env.v_step(states, actions, key)
    obs.block_until_ready()


def run_vmapped_scan(env, params, batch_size, n_steps, repeats=1):
    rng = jax.random.PRNGKey(1)

    def policy_step(state_carry, tmp):
        obs, states, rng = state_carry
        if params is None:
            actions = jnp.zeros((batch_size, env.n_agents, env.action_space.shape[0]))
        else:
            actions = params.apply_fn(params.params, obs)
        rng, key = jax.random.split(rng, 2)
        key = jax.random.split(key, batch_size)
        obs, states, r, a, d = env.v_step(states, actions, key)
        carry = [obs, states, rng]
        return carry, []

    # Scan over episode step loop
    for _ in range(repeats):
        rng, key = jax.random.split(rng, 2)
        key = jax.random.split(key, batch_size)
        obs, states = env.v_reset(key)
        _, scan_out = jax.lax.scan(policy_step, [obs, states, rng], [], length=n_steps)
    scan_out[1].block_until_ready()


def run_single_no_scan(env, params, n_steps, repeats=1):
    rng = jax.random.PRNGKey(1)
    for _ in range(repeats):
        rng, key = jax.random.split(rng, 2)
        obs, states = env.reset(key)
        for i in range(n_steps):
            rng, key = jax.random.split(rng, 2)
            if params is None:
                actions = jnp.zeros((env.n_agents, env.action_space.shape[0]))
            else:
                actions = params.apply_fn(params.params, obs)
            obs, states, r, a, d = env.step(states, actions, key)
    obs.block_until_ready()


def run_single_scan(env, params, n_steps, repeats=1):
    rng = jax.random.PRNGKey(1)

    def policy_step(state_carry, tmp):
        obs, states, rng = state_carry
        if params is None:
            actions = jnp.zeros((env.n_agents, env.action_space.shape[0]))
        else:
            actions = params.apply_fn(params.params, obs)
        rng, key = jax.random.split(rng, 2)
        obs, states, r, a, d = env.step(states, actions, key)
        carry = [obs, states, rng]
        return carry, []

    # Scan over episode step loop
    for _ in range(repeats):
        rng, key = jax.random.split(rng, 2)
        obs, states = env.reset(key)
        _, scan_out = jax.lax.scan(policy_step, [obs, states, rng], [], length=n_steps)

    scan_out[1].block_until_ready()


def to_human_readable(step_per_second):
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
    if step_per_second > 100:
        step_per_second_str = f"{step_per_second:0.0f}{postfix}"
    else:
        step_per_second_str = f"{step_per_second:0.1f}{postfix}"
    sec_to_100M = 100_000_000 / orig_step_per_second
    if sec_to_100M >= 120 * 60:  # 120 minutes
        hours = sec_to_100M / 60 / 60
        time_to_100m = f"{int(round(hours))} hours"
    elif sec_to_100M >= 60:
        mins = sec_to_100M / 60
        time_to_100m = f"{int(round(mins))} minutes"
    else:
        time_to_100m = f"{sec_to_100M:0.1f} seconds"
    return step_per_second_str, time_to_100m


class ConvPolicy(nn.Module):
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]
        five_dims = False
        if len(x.shape) == 5:
            five_dims = True
            x = jnp.reshape(x, (-1, *x.shape[-3:]))
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
        x = nn.Conv(
            features=256,
            kernel_size=(5, 5),
            strides=(2, 2),
            name="conv5",
        )(x)
        x = nn.relu(x)
        # global average pooling
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(features=128, name="hidden")(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_outputs, name="logits")(x)
        if five_dims:
            x = jnp.reshape(x, (batch_size, -1, self.num_outputs))
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
    BATCH_SIZES = [
        1,
        8,
        32,
        128,
        512,
        2048,
        8192,
        8192 * 4,
        8191 * 4 * 4,
        8191 * 4 * 4 * 4,
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--format", default="latex", type=str, help="Output format")
    parser.add_argument("--no_policy", action="store_true", help="Don't run policy")
    args = parser.parse_args()

    latex_format = args.format == "latex"
    if latex_format:
        print("\\begin{tabular}{lc|cccc}\\toprule")
        print(
            "Device & Batch size & \\multicolumn{2}{c}{Vector} & \\multicolumn{2}{c}{RGB} \\\\"
        )
        print("& & steps/s & Time to 100M & steps/s & Time to 100M \\\\\\midrule")
    for batch_size in BATCH_SIZES:
        if latex_format:
            print(f" DEVICE &  {batch_size}", end="")
        for obs_type in ["vector", "rgb"]:
            env = GigastepEnv(n_agents=4, obs_type=obs_type)
            if obs_type == "vector":
                policy = MLPPolicy(num_outputs=env.action_space.shape[0])
                input_shape = (1, env.observation_space.shape[0])
            else:
                policy = ConvPolicy(num_outputs=env.action_space.shape[0])
                input_shape = (1, 84, 84, 3)
            nn_state = create_train_state(policy, jax.random.PRNGKey(0), input_shape)
            if args.no_policy:
                nn_state = None

            start_time = time.time()
            if batch_size > 8192 and obs_type == "rgb":
                time.sleep(0.5)
            else:
                if batch_size == 1:
                    run_single_no_scan(env, nn_state, n_steps=2000, repeats=5)
                else:
                    run_vmapped_no_scan(
                        env, nn_state, n_steps=2000, repeats=5, batch_size=batch_size
                    )
            elapsed = time.time() - start_time
            num_agent_steps = 2000 * batch_size * 4 * 5
            step_per_second = num_agent_steps / elapsed
            step_per_second_str, time_to_100m = to_human_readable(step_per_second)

            if latex_format:
                print(f" & {step_per_second_str} & {time_to_100m} ", end="")

            if not latex_format:
                print(
                    f"{batch_size} {obs_type} steps/s: {step_per_second_str} time to 100M: {time_to_100m}"
                )
        if latex_format:
            print(" \\\\")

    if latex_format:
        print("\\bottomrule")
        print("\\end{tabular}")


if __name__ == "__main__":
    main()