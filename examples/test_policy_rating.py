
import argparse
import time

import jax
import jax.numpy as jnp
from gigastep import GigastepEnv
import time
import torch
from gigastep.rate_policy import Rating

def main():
    class Cfg():
        def __init__(self):
            pass
    env_cfg = Cfg()
    env_cfg.reward_game_won = 0
    env_cfg.reward_defeat_one_opponent = 0
    env_cfg.reward_detection = 0
    env_cfg.reward_idle = 0
    env_cfg.reward_collision = 0
    env_cfg.reward_damage = 1
    import jax.numpy as jnp
    env_cfg.cone_depth = 15.0
    env_cfg.cone_angle = jnp.pi * 1.99
    env_cfg.enable_waypoints = False
    env_cfg.use_stochastic_obs = False
    env_cfg.use_stochastic_comm = False
    env_cfg.max_agent_in_vec_obs = 40
    env_cfg.resolution_x = 40
    env_cfg.resolution_y = 40

    device = torch.device("cuda:0")
    save_path = "../evaluation/trained_models/"
    env_name = "identical_2_vs_2"

    rate = Rating(device, save_path, env_name, env_cfg, num_match= 3)
    actor_critic,_ = torch.load("../evaluation/trained_models/ppogigastepgiga20230521-144137.667049.pt",
                            map_location=torch.device('cpu'))

    rate_0 = rate.rate_policy(actor_critic)
    print(f"The uploaded policy got rating {rate_0}")
    ## after the pool has certain members
    rate.update_pool_rating()


if __name__ == "__main__":
    main()


