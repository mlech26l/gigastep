import sys
import time

import PIL.Image
import numpy as np
import pyhopper
import jax
import jax.numpy as jnp

from gigastep import make_scenario, GigastepViewer
from gigastep.evaluator import Evaluator
import seaborn as sns
import matplotlib.pyplot as plt

SLEEP_TIME = 0.01
from PIL import Image


def circle_vs_straight(env):
    evaluator = Evaluator(env)
    viewer = GigastepViewer(84 * 4, show_num_agents=0)

    rng = jax.random.PRNGKey(3)
    img_list = []
    while True:
        ep_done = False
        key, rng = jax.random.split(rng, 2)
        obs, state = env.reset(key)
        while not ep_done:
            rng, key, key2 = jax.random.split(rng, 3)
            action_ego = jnp.zeros((env.n_agents, 3))
            # action_opp = jnp.repeat(jnp.array([[1.0, 0.0, 00]]), env.n_agents, axis=0)
            action_opp = jnp.zeros((env.n_agents, 3))

            action = evaluator.merge_actions(action_ego, action_opp)

            rng, key = jax.random.split(rng, 2)
            obs, state, r, dones, ep_done = env.step(state, action, key)
            evaluator.update_step(r, dones, ep_done)

            img = viewer.draw(env, state, obs)
            # img_list.append(PIL.Image.fromarray(img))
            # if len(img_list) == 400:
            #     # create gif
            #     img_list[0].save(
            #         "circle_vs_straight.gif",
            #         save_all=True,
            #         append_images=img_list[1:],
            #         optimize=False,
            #         duration=20,
            #         loop=0,
            #     )
            #     sys.exit(1)
            if viewer.should_pause:
                while True:
                    img = viewer.draw(env, state, obs)
                    time.sleep(SLEEP_TIME)
                    if viewer.should_pause:
                        break
            if viewer.should_quit:
                sys.exit(1)
            time.sleep(SLEEP_TIME)
        evaluator.update_episode()
        print(str(evaluator))
        # if frame_idx > 400:
        #     sys.exit(1)


def replay_policy(env, params):
    evaluator = Evaluator(env)
    viewer = GigastepViewer(84 * 4, show_num_agents=0)
    params = jax.tree_map(lambda x: jnp.array(x), params)

    rng = jax.random.PRNGKey(3)

    for opponent in evaluator.policies:
        print("Opponent", str(opponent))
        for ep_idx in range(10):
            ep_done = False
            key, rng = jax.random.split(rng, 2)
            state, obs = env.reset(key)
            while not ep_done:
                rng, key, key2 = jax.random.split(rng, 3)
                x1 = jax.nn.tanh(jnp.dot(obs, params["w1"]) + params["b1"])
                action_ego = jax.nn.tanh(jnp.dot(x1, params["w2"]) + params["b2"])
                # action_ego = evaluator.policies[-1].apply(obs, key2)

                action_opp = opponent.apply(obs, key2)

                action = evaluator.merge_actions(action_ego, action_opp)

                rng, key = jax.random.split(rng, 2)
                obs, state, r, dones, ep_done = env.step(state, action, key)
                evaluator.update_step(r, dones, ep_done)

                img = viewer.draw(env, state, obs)
                if viewer.should_pause:
                    while True:
                        img = viewer.draw(env, state, obs)
                        time.sleep(SLEEP_TIME)
                        if viewer.should_pause:
                            break
                if viewer.should_quit:
                    sys.exit(1)
                time.sleep(SLEEP_TIME)
            evaluator.update_episode()
            print(str(evaluator))
            # if frame_idx > 400:
            #     sys.exit(1)


def run_n_steps2(params, env):
    evaluator = Evaluator(env)

    batch_size = 128
    n_steps = 1000
    # opponent = evaluator.policies[-1]

    rng = jax.random.PRNGKey(2)
    rng, key = jax.random.split(rng, 2)
    key = jax.random.split(key, batch_size)
    states, obs = env.v_reset(key)

    params = jax.tree_map(lambda x: jnp.array(x), params)
    n_agents = states[0]["x"].shape[1]
    reward = jnp.zeros((batch_size, n_agents))

    def policy_step(state_carry, tmp):
        states, obs, rng, reward = state_carry
        rng, key, key2 = jax.random.split(rng, 3)
        x1 = jax.nn.tanh(jnp.dot(obs, params["w1"]) + params["b1"])
        action_ego = jax.nn.tanh(jnp.dot(x1, params["w2"]) + params["b2"])

        key2 = jax.random.split(key2, batch_size)
        # action_ego = evaluator.policies[-1].v_apply(obs, key2)
        # action_opp = evaluator.policies[0].v_apply(obs, key2)
        action_opp = evaluator.policies[-2].v_apply(obs, key2)
        # action_opp = opponent.v_apply(obs, key2)
        # action_opp = jnp.zeros_like(action_ego)

        actions = evaluator.v_merge_actions(action_ego, action_opp)
        rng, key = jax.random.split(rng, 2)
        key = jax.random.split(key, batch_size)
        states, obs, r, a, d = env.v_step(states, actions, key)

        reward = reward + r
        carry = [states, obs, rng, reward]
        return carry, []

    def policy_step2(state_carry, tmp):
        states, obs, rng, reward = state_carry
        rng, key, key2 = jax.random.split(rng, 3)
        x1 = jax.nn.tanh(jnp.dot(obs, params["w1"]) + params["b1"])
        action_ego = jax.nn.tanh(jnp.dot(x1, params["w2"]) + params["b2"])

        key2 = jax.random.split(key2, batch_size)
        # action_ego = evaluator.policies[-1].v_apply(obs, key2)
        # action_opp = jax.nn.tanh(evaluator.policies[0].v_apply(obs, key2))
        action_opp = evaluator.policies[-1].v_apply(obs, key2)
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

    rng = jax.random.PRNGKey(3)
    rng, key = jax.random.split(rng, 2)
    key = jax.random.split(key, batch_size)
    states, obs = env.v_reset(key)
    reward2 = jnp.zeros((batch_size, n_agents))
    carry_out2, scan_out = jax.lax.scan(
        policy_step2, [states, obs, rng, reward2], [], length=n_steps
    )
    # print("total agg", carry_out[-1].sum())
    reward2 = carry_out2[-1] * (env.teams[None, :] == 0)
    reward2 = 2 * reward2.sum(axis=1).mean()
    # reward_other = carry_out[-1] * (env.teams[None, :] == 1)
    # reward_other = reward_other.sum(axis=1).mean()
    # print("reward", reward, "reward_other", reward_other)
    return reward + reward2


if __name__ == "__main__":
    env = make_scenario(
        "identical_20_vs_20",
        obs_type="vector",
        reward_game_won=1,
        reward_defeat_one_opponent=1,
        reward_detection=0,
        reward_damage=1,
        reward_idle=0,
        reward_agent_disabled=1,
        reward_collision=0,
        use_stochastic_obs=False,
        use_stochastic_comm=False,
    )
    params = {
        "w1": jnp.zeros((env.observation_space.shape[0], 10)),
        "b1": jnp.zeros(10),
        "w2": jnp.zeros((10, env.action_space.shape[0])),
        "b2": jnp.zeros(env.action_space.shape[0]),
    }
    # params = np.load("best_params.npz")
    # params = {k: params[k] for k in params.files}
    # replay_policy(env, params)

    # circle_vs_straight(env)
    # import sys
    #
    # sys.exit()
    r = run_n_steps2(params, env)
    print(r)

    hidden = 64
    search = pyhopper.Search(
        w1=pyhopper.float(shape=(env.observation_space.shape[0], hidden)),
        b1=pyhopper.float(shape=(hidden,)),
        w2=pyhopper.float(shape=(hidden, env.action_space.shape[0])),
        b2=pyhopper.float(shape=(env.action_space.shape[0],)),
    )
    best_params = search.run(run_n_steps2, "maximize", "220min", kwargs={"env": env})
    print(f"Ran a total of {search.history.steps[-1]*128*2*200/10e6:0.1f}M steps.")
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

    replay_policy(env, best_params)
    np.savez("best_params.npz", **best_params)