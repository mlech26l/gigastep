import argparse
import time

import jax
import jax.numpy as jnp
from gigastep import GigastepEnv
import time
import time


def apply_policy(params, obs):
    x1 = jax.nn.tanh(jnp.dot(obs, params["w1"]) + params["b1"])
    action = jax.nn.tanh(jnp.dot(x1, params["w2"]) + params["b2"])
    return action


def run_vmapped_no_scan(env, params, batch_size, n_steps, repeats=1):
    rng = jax.random.PRNGKey(1)
    for _ in range(repeats):
        rng, key = jax.random.split(rng, 2)
        key = jax.random.split(key, batch_size)
        states, obs = env.v_reset(key)
        for i in range(n_steps):
            rng, key = jax.random.split(rng, 2)
            actions = apply_policy(params, obs)
            # actions = jnp.zeros((batch_size, env.n_agents, env.action_space.shape[0]))
            key = jax.random.split(key, batch_size)
            states, obs, r, a, d = env.v_step(states, actions, key)
    obs.block_until_ready()


def run_vmapped_scan(env, params, batch_size, n_steps, repeats=1):
    rng = jax.random.PRNGKey(1)

    def policy_step(state_carry, tmp):
        states, obs, rng = state_carry
        actions = apply_policy(params, obs)
        # actions = jnp.zeros((batch_size, env.n_agents, env.action_space.shape[0]))
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
            actions = apply_policy(params, obs)
            # actions = jnp.zeros((env.n_agents, env.action_space.shape[0]))
            states, obs, r, a, d = env.step(states, actions, key)
    obs.block_until_ready()


def run_single_scan(env, params, n_steps, repeats=1):
    rng = jax.random.PRNGKey(1)

    def policy_step(state_carry, tmp):
        states, obs, rng = state_carry
        actions = apply_policy(params, obs)
        # actions = jnp.zeros((env.n_agents, env.action_space.shape[0]))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4096 * 16, type=int)  # in minutes
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
    params = {
        "w1": jax.random.normal(
            jax.random.PRNGKey(1), shape=(env.observation_space.shape[0], args.hidden)
        ),
        "b1": jax.random.normal(jax.random.PRNGKey(2), shape=(args.hidden,)),
        "w2": jax.random.normal(
            jax.random.PRNGKey(3), shape=(args.hidden, env.action_space.shape[0])
        ),
        "b2": jax.random.normal(
            jax.random.PRNGKey(4), shape=(env.action_space.shape[0],)
        ),
    }
    with GigastepTimer(
        "single scan", args.n_steps * args.repeats, 1, args.n_agents, args.obs_type
    ):
        run_single_scan(env, params, args.n_steps, args.repeats)
    with GigastepTimer(
        "single no scan", args.n_steps * args.repeats, 1, args.n_agents, args.obs_type
    ):
        run_single_no_scan(env, params, args.n_steps, args.repeats)
    with GigastepTimer(
        "vmapped scan",
        args.n_steps,
        args.batch_size * args.repeats,
        args.n_agents,
        args.obs_type,
    ):
        run_vmapped_scan(
            env, params, args.n_steps * args.repeats, args.batch_size, args.repeats
        )
    with GigastepTimer(
        "vmapped no scan",
        args.n_steps,
        args.batch_size * args.repeats,
        args.n_agents,
        args.obs_type,
    ):
        run_vmapped_no_scan(
            env, params, args.n_steps * args.repeats, args.batch_size, args.repeats
        )


if __name__ == "__main__":
    main()