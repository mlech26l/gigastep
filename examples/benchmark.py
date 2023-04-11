import argparse
import time

import jax
import jax.numpy as jnp
from spacesnakx import Spacesnakx
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
    if step_per_second > 1e9:
        postfix = "G"
        step_per_second = step_per_second / 1e9
    elif step_per_second > 1e6:
        postfix = "M"
        step_per_second = step_per_second / 1e6
    elif step_per_second > 1e3:
        postfix = "k"
        step_per_second = step_per_second / 1e3
    print(f"{step_per_second:0.1f}{postfix} steps per second")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)  # in minutes
    parser.add_argument("--n_agents", default=8, type=int)  # in minutes
    parser.add_argument("--n_steps", default=100, type=int)  # in minutes
    args = parser.parse_args()
    env = Spacesnakx(n_agents=args.n_agents)
    run_n_steps(env, args.batch_size, 10)

    start_time = time.time()
    run_n_steps2(env, args.batch_size, args.n_steps)
    # run_n_steps(env, args.batch_size, args.n_steps)
    end_time = time.time()
    num_agent_steps = args.n_steps * args.batch_size * args.n_agents
    steps_per_seconds = num_agent_steps / (end_time - start_time)
    print(f"Run {num_agent_steps} agent steps in {end_time - start_time} seconds")
    print_step_per_second(steps_per_seconds)


if __name__ == "__main__":
    main()