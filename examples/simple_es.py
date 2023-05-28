import numpy as np
import pyhopper
import jax
import jax.numpy as jnp

from gigastep import make_scenario
from gigastep.evaluator import Evaluator
import seaborn as sns
import matplotlib.pyplot as plt


def run_n_steps2(params, env):
    evaluator = Evaluator(env)

    batch_size = 16
    n_steps = 300
    opponent = evaluator.policies[-1]

    rng = jax.random.PRNGKey(2)
    rng, key = jax.random.split(rng, 2)
    key = jax.random.split(key, batch_size)
    states, obs = env.v_reset(key)
    states = env.v_set_aux_reward_factor(states, 0.0)

    params = jax.tree_map(lambda x: jnp.array(x), params)
    n_agents = states[0]["x"].shape[1]
    reward = jnp.zeros((batch_size, n_agents))

    def policy_step(state_carry, tmp):
        states, obs, rng, reward = state_carry
        rng, key, key2 = jax.random.split(rng, 3)
        x1 = jax.nn.tanh(jnp.dot(obs, params["w1"]) + params["b1"])
        action_ego = jnp.dot(x1, params["w2"]) + params["b2"]

        key2 = jax.random.split(key2, batch_size)
        action_opp = opponent.v_apply(obs, key2)
        # action_opp = jnp.zeros_like(action_ego)

        actions = evaluator.v_merge_actions(action_ego, action_opp)
        rng, key = jax.random.split(rng, 2)
        key = jax.random.split(key, batch_size)
        states, obs, r, a, d = env.v_step(states, actions, key)

        reward = reward + r
        carry = [states, obs, rng, reward]
        return carry, []

    # Scan over episode step loop
    carry_out, scan_out = jax.lax.scan(
        policy_step, [states, obs, rng, reward], [], length=n_steps
    )
    # print("total agg", carry_out[-1].sum())
    reward = carry_out[-1] * (env.teams[None, :] == 0)
    reward = reward.sum(axis=1).mean()
    return reward


if __name__ == "__main__":
    env = make_scenario("identical_20_vs_20", obs_type="vector")
    params = {
        "w1": jnp.zeros((env.observation_space.shape[0], 10)),
        "b1": jnp.zeros(10),
        "w2": jnp.zeros((10, env.action_space.shape[0])),
        "b2": jnp.zeros(env.action_space.shape[0]),
    }
    r = run_n_steps2(params, env)
    print(r)

    hidden = 32
    search = pyhopper.Search(
        w1=pyhopper.float(shape=(env.observation_space.shape[0], hidden)),
        b1=pyhopper.float(shape=(hidden,)),
        w2=pyhopper.float(shape=(hidden, env.action_space.shape[0])),
        b2=pyhopper.float(shape=(env.action_space.shape[0],)),
    )
    best_params = search.run(run_n_steps2, "maximize", "2min", kwargs={"env": env})

    sns.set()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(
        search.history.steps, search.history.best_fs, color="red", label="Best so far"
    )
    ax.scatter(
        x=search.history.steps, y=search.history.fs, color="blue", label="Evaluated"
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Objective value")
    fig.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("search.png")
    plt.close(fig)