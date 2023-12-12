import os
import sys
import time

import cv2
import jax
import numpy as np

from gigastep import GigastepViewer, GigastepEnv, make_scenario, ScenarioBuilder
from gigastep.evaluator import Evaluator
import jax.numpy as jnp

SLEEP_TIME = 0.01

import torch


def get_action_policy(actor_critic,
                      obs,
                      recurrent_hidden_states,
                      discrete_actions,
                      n_ego_agent,
                      device,
                      vectorize=False):
    obs = torch.tensor(np.asarray(obs), device=device)

    if not vectorize:
        obs = torch.unsqueeze(obs, 0)
    obs = torch.moveaxis(obs, -1, 2)
    obs = obs[:, :n_ego_agent, ::]
    obs = obs.float().contiguous()
    masks = torch.ones(1, 1).to(device)
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs.float(), recurrent_hidden_states, masks, deterministic=True)
    if not vectorize:
        action = torch.squeeze(action, 0)

    if discrete_actions:
        action = jnp.array(action.detach().cpu().numpy().astype(np.int32))
    else:
        action = jnp.array(action.detach().cpu().numpy().astype(np.float32))
    return action, recurrent_hidden_states


def loop_env(env, policy = None, device = "cpu", headless = False):
    evaluator = Evaluator(env)
    viewer = GigastepViewer(84 * 2, show_num_agents=env.n_agents)
    rng = jax.random.PRNGKey(3)

    for opponent in evaluator.policies:
        print("Opponent", str(opponent))
        for ep_idx in range(5):
            ep_done = False
            key, rng = jax.random.split(rng, 2)
            obs, state = env.reset(key)
            while not ep_done:
                rng, key, key2 = jax.random.split(rng, 3)
                if policy is None:
                    action_ego = jnp.zeros(
                        (env.n_agents, 3)
                    )  # ego does nothing
                else:
                    action, recurrent_hidden_states = get_action_policy(policy,
                                                                 obs,
                                                                 recurrent_hidden_states,
                                                                 env.discrete_actions,
                                                                 env.n_teams[0],
                                                                 device = torch.device(device))

                action_opp = opponent.apply(obs, key2)

                action = evaluator.merge_actions(action_ego, action_opp)

                rng, key = jax.random.split(rng, 2)
                obs, state, r, dones, ep_done = env.step(state, action, key)
                evaluator.update_step(r, dones, ep_done)
                if not headless:
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

    return [evaluator.team_a_wins / evaluator.total_games,
                    evaluator.team_b_wins / evaluator.total_games]





def loop_env_vectorized(env, policy = None, device = "cpu"):
    evaluator = Evaluator(env)
    batch_size = 20
    rng = jax.random.PRNGKey(3)

    for opponent in evaluator.policies:
        print("Opponent", str(opponent))
        for ep_idx in range(5):
            ep_done = np.zeros(batch_size, dtype=jnp.bool_)
            key, rng = jax.random.split(rng, 2)
            key = jax.random.split(key, batch_size)
            obs, state = env.v_reset(key)
            t = 0
            
            recurrent_hidden_states = torch.zeros(env.n_teams[0], 
                                                  policy.recurrent_hidden_state_size,
                                                  device=device) if policy is not None else None
            while not jnp.all(ep_done):
                rng, key, key2 = jax.random.split(rng, 3)
                if policy is None:
                    action_ego = jnp.zeros(
                        (batch_size, env.n_agents, 3)
                    )  # ego does nothing
                else:
                    action, recurrent_hidden_states = get_action_policy(policy,
                                                                 obs,
                                                                 recurrent_hidden_states,
                                                                 env.discrete_actions,
                                                                 env.n_teams[0],
                                                                 device = torch.device(device))

                key2 = jax.random.split(key2, batch_size)
                action_opp = opponent.v_apply(obs, key2)

                action = evaluator.v_merge_actions(action_ego, action_opp)

                rng, key = jax.random.split(rng, 2)
                key = jax.random.split(key, batch_size)
                obs, state, r, dones, ep_done = env.v_step(state, action, key)
                evaluator.update_step(r, dones, ep_done)

                time.sleep(SLEEP_TIME)
                t += 1
                # print("t", t, "ep_done", ep_done)
            evaluator.update_episode()
            print(str(evaluator))
            # if frame_idx > 400:
            #     sys.exit(1)


    return [evaluator.team_a_wins/evaluator.total_games,
                    evaluator.team_b_wins/evaluator.total_games]
            



if __name__ == "__main__":
    # convert -delay 3 -loop 0 video/scenario/frame_*.png video/scenario.webp
    wining_rate_vec = loop_env_vectorized(
        env=make_scenario("identical_20_vs_20", use_stochastic_comm=False)
    )
    wining_rate = loop_env(env=make_scenario("identical_20_vs_20", use_stochastic_comm=False))

    print("wining_rate_vec", wining_rate_vec)
    print("wining_rate", wining_rate)
