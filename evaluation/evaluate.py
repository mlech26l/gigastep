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

    action = action.squeeze(-1)

    if discrete_actions:
        action = jnp.array(action.detach().cpu().numpy().astype(np.int32))
    else:
        action = jnp.array(action.detach().cpu().numpy().astype(np.float32))
    return action, recurrent_hidden_states


def loop_env(env, policy = None, device = "cpu", headless = False):
    evaluator = Evaluator(env)
    viewer = GigastepViewer(frame_size=84*2, show_num_agents=0 if env.discrete_actions else env.n_agents)

    rng = jax.random.PRNGKey(3)

    for opponent in evaluator.policies:
        print("Opponent", str(opponent))
        for ep_idx in range(5):
            ep_done = False
            key, rng = jax.random.split(rng, 2)
            state, obs = env.reset(key)
            recurrent_hidden_states = torch.zeros(env.n_agents,
                                                  policy.recurrent_hidden_state_size,
                                                  device=device) if policy is not None else None
            while not ep_done:
                rng, key, key2 = jax.random.split(rng, 3)
                if policy is None:
                    action_ego = jnp.zeros(
                        (env.n_agents, 3)
                    )  # ego does nothing
                else:
                    action_ego, recurrent_hidden_states = get_action_policy(policy,
                                                            obs,
                                                            recurrent_hidden_states,
                                                            env.discrete_actions,
                                                            env.n_agents,
                                                            device = torch.device(device),
                                                            vectorize = False
                    )

                action_opp = opponent.apply(obs, key2)

                action = evaluator.merge_actions(action_ego, action_opp)

                rng, key = jax.random.split(rng, 2)
                state, obs, r, dones, ep_done = env.step(state, action, key)
                evaluator.update_step(r, dones, ep_done)
                if not headless:
                    img = viewer.draw(env, state, obs)
                    # if viewer.should_pause:
                    #     while True:
                    #         img = viewer.draw(env, state, obs)
                    #         time.sleep(SLEEP_TIME)
                    #         if viewer.should_pause:
                    #             break
                    if viewer.should_quit:
                        sys.exit(1)
                time.sleep(SLEEP_TIME)
            evaluator.update_episode()
            print(str(evaluator))
            # if frame_idx > 400:
            #     sys.exit(1)
        break
    return [evaluator.team_a_wins / evaluator.total_games,
                    evaluator.team_b_wins / evaluator.total_games]





def loop_env_vectorized(env, policy = None, device = "cpu"):
    evaluator = Evaluator(env)
    batch_size = 20
    rng = jax.random.PRNGKey(3)

    for opponent in evaluator.policies:
        print("Opponent", str(opponent))
        for ep_idx in range(1):
            ep_done = np.zeros(batch_size, dtype=jnp.bool_)
            key, rng = jax.random.split(rng, 2)
            key = jax.random.split(key, batch_size)
            state, obs = env.v_reset(key)
            t = 0
            
            recurrent_hidden_states = torch.zeros(env.n_agents,
                                                  policy.recurrent_hidden_state_size,
                                                  device=device) if policy is not None else None
            while not jnp.all(ep_done):
                rng, key, key2 = jax.random.split(rng, 3)
                if policy is None:
                    action_ego = jnp.zeros(
                        (batch_size, env.n_agents, 3)
                    )  # ego does nothing
                else:
                    action_ego, recurrent_hidden_states = get_action_policy(policy,
                                                                 obs,
                                                                 recurrent_hidden_states,
                                                                 env.discrete_actions,
                                                                 env.n_agents,
                                                                 device = torch.device(device),
                                                                 vectorize = True)

                key2 = jax.random.split(key2, batch_size)
                action_opp = opponent.v_apply(obs, key2)

                action = evaluator.v_merge_actions(action_ego, action_opp)

                rng, key = jax.random.split(rng, 2)
                key = jax.random.split(key, batch_size)
                state, obs, r, dones, ep_done = env.v_step(state, action, key)
                evaluator.update_step(r, dones, ep_done)

                time.sleep(SLEEP_TIME)
                t += 1
                # print("t", t, "ep_done", ep_done)
            evaluator.update_episode()
            print(str(evaluator))
            # if frame_idx > 400:
            #     sys.exit(1)
        break

    return [evaluator.team_a_wins/evaluator.total_games,
                    evaluator.team_b_wins/evaluator.total_games]
            



if __name__ == "__main__":
    # convert -delay 3 -loop 0 video/scenario/frame_*.png video/scenario.webp

    trained_agent_dict = {
        "identical_2_vs_2": {
            "discrete_actions": True,
            "obs_type": "vector",
            "path": "trained_models/ppogigastepgiga20230523-173609.588417.pt",
            "log": "Simultaneouslyselfplay-05/23/23-162638.720489"
        },
        "identical_5_vs_5": {
            "discrete_actions": True,
            "obs_type": "vector",
            "path": "trained_models/ppogigastepgiga20230523-165316.935000.pt",
            "log": "multaneouslyselfplay-05/23/23-162638.720489"
        },
        "identical_10_vs_10": {
            "discrete_actions": True,
            "obs_type": "vector",
            "path": "trained_models/ppogigastepgiga20230523-164935.795041.pt",
            "log": "Simultaneouslyselfplay-05/23/23-161554.854292"
        },
        "identical_20_vs_20": {
            "discrete_actions": True,
            "obs_type": "vector",
            "path": "trained_models/ppogigastepgiga20230523-181621.808298.pt",
            "log": "Simultaneouslyselfplay-05/23/23-165237.120339"
        },
        "identical_20_vs_5": {
            "discrete_actions": True,
            "obs_type": "vector",
            "path": "trained_models/ppogigastepgiga20230523-175305.597032.pt",
            "log": "Simultaneouslyselfplay-05/23/23-171518.435491"
        },
        "special_10_vs_10": {
            "discrete_actions": True,
            "obs_type": "vector",
            "path": "trained_models/ppogigastepgiga20230523-192651.395839.pt",
            "log": "Simultaneouslyselfplay-05/23/23-185108.873903"
        },
        "special_5_vs_5": {
            "discrete_actions": True,
            "obs_type": "vector",
            "path": "trained_models/ppogigastepgiga20230523-174424.987372.pt",
            "log": "Simultaneouslyselfplay-05/23/23-171518.309974"
        },
        "special_5_vs_1": {
            "discrete_actions": True,
            "obs_type": "vector",
            "path": "trained_models/ppogigastepgiga20230523-174313.936525.pt",
            "log": "Simultaneouslyselfplay-05/23/23-171518.573258"
        },
        "identical_5_vs_1": {
            "discrete_actions": True,
            "obs_type": "vector",
            "path": "trained_models/ppogigastepgiga20230523-173639.001829.pt",
            "log": "Simultaneouslyselfplay-05/23/23-171519.037673"
        },
        "special_20_vs_5": {
            "discrete_actions": True,
            "obs_type": "vector",
            "path": "trained_models/ppogigastepgiga20230523-175650.084486.pt",
            "log": "Simultaneouslyselfplay-05/23/23-171518.373156"
        }
    }
    
    for env_name, load_config in trained_agent_dict.items():

        ## env configurations
        class Cfg():
            def __init__(self):
                pass
        env_cfg = Cfg()
        env_cfg.reward_game_won = 10
        env_cfg.reward_defeat_one_opponent = 100
        env_cfg.reward_detection = 0
        env_cfg.reward_idle = 0
        env_cfg.reward_collision = 0
        env_cfg.reward_damage = 0
        import jax.numpy as jnp
        env_cfg.cone_depth = 15.0
        env_cfg.cone_angle = jnp.pi * 1.99
        env_cfg.enable_waypoints = False
        env_cfg.use_stochastic_obs = False
        env_cfg.use_stochastic_comm = False
        env_cfg.max_agent_in_vec_obs = 100
        env_cfg.resolution_x = 84
        env_cfg.resolution_y = 84
        env_cfg.discrete_actions = True
        env_cfg.obs_type = load_config["obs_type"]

        ## create env
        env = make_scenario(env_name,
                            **vars(env_cfg)
                            )


        print(f"env_name: {env_name}")
        actor_critic,_ = torch.load(
            load_config["path"],
            map_location=torch.device('cpu')
        )
        device = torch.device("cuda:0")
        actor_critic.to(device)

        wining_rate_vec = loop_env_vectorized(
            env=env,
            policy=actor_critic,
            device = "cuda:0"
        )
        # wining_rate = loop_env(
        #     env=env,
        #     policy=actor_critic,
        #     device = "cuda:0",
        #     headless = False
        # )
