import argparse
import time

import jax
import jax.numpy as jnp
from gigastep import GigastepEnv
import time
import time


def run_n_steps(env, batch_size, n_steps):
    rng = jax.random.PRNGKey(1)
    rng, key = jax.random.split(rng, 2)
    key = jax.random.split(key, batch_size)
    states, obs = env.v_reset(key)
    n_agents = states[0]["x"].shape[1]
    for i in range(n_steps):
        rng, key = jax.random.split(rng, 2)

        actions = jax.random.uniform(
            key, shape=(batch_size, n_agents, 3), minval=-1, maxval=1
        )
        rng, key = jax.random.split(rng, 2)
        key = jax.random.split(key, batch_size)
        states, obs, r, a, d = env.v_step(states, actions, key)


def run_n_steps2(env, batch_size, n_steps):
    rng = jax.random.PRNGKey(1)
    rng, key = jax.random.split(rng, 2)
    key = jax.random.split(key, batch_size)
    states, obs = env.v_reset(key)
    n_agents = states[0]["x"].shape[1]

    def policy_step(state_carry, tmp):
        states, rng = state_carry
        rng, key = jax.random.split(rng, 2)
        actions = jax.random.uniform(
            key, shape=(batch_size, n_agents, 3), minval=-1, maxval=1
        )
        rng, key = jax.random.split(rng, 2)
        key = jax.random.split(key, batch_size)
        states, obs, r, a, d = env.v_step(states, actions, key)

        carry = [states, rng]
        return carry, []

    # Scan over episode step loop
    _, scan_out = jax.lax.scan(policy_step, [states, rng], [], length=n_steps)


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
    if sec_to_100M >= 60:
        print(f" (time to 100M steps: {sec_to_100M/60:0.1f} minutes)", end="")
    else:
        print(f" (time to 100M steps: {sec_to_100M:0.1f} seconds)", end="")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1024, type=int)  # in minutes
    parser.add_argument("--n_agents", default=8, type=int)  # in minutes
    parser.add_argument("--n_steps", default=300, type=int)  # in minutes
    parser.add_argument("--obs_type", default="rgb")  # in minutes
    args = parser.parse_args()
    env = GigastepEnv(n_agents=args.n_agents, obs_type=args.obs_type)
    run_n_steps(env, args.batch_size, 10)

    nojit_start = time.time()
    run_n_steps(env, args.batch_size, args.n_steps)
    nojit_time = time.time()
    jit_start = time.time()
    run_n_steps2(env, args.batch_size, args.n_steps)
    jit_time = time.time()
    num_agent_steps = args.n_steps * args.batch_size * args.n_agents
    steps_per_seconds_nojit = num_agent_steps / (nojit_time - nojit_start)
    steps_per_seconds_jit = num_agent_steps / (jit_time - jit_start)
    print(
        f"Run {num_agent_steps:.0f} agent steps in {nojit_time - nojit_start:.1f} seconds (no jit)"
    )
    print_step_per_second(steps_per_seconds_nojit)
    print(
        f"Run {num_agent_steps:.0f} agent steps in {jit_time - jit_start:.1f} seconds (jit)"
    )
    print_step_per_second(steps_per_seconds_jit)


if __name__ == "__main__":
    main()