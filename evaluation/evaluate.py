import os
import sys
import time

import cv2
import jax
import numpy as np

from gigastep import GigastepViewer, GigastepEnv, make_scenario, ScenarioBuilder
from gigastep.evaluator import Evaluator, loop_env_vectorized,loop_env
import jax.numpy as jnp

import torch



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
