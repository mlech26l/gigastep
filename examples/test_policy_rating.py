
import argparse
import time

import jax
import jax.numpy as jnp
from gigastep import GigastepEnv
import time
import torch
from gigastep.evaluator import Rating

def main():
    class Cfg():
        def __init__(self):
            pass
    env_cfg = Cfg()
    env_cfg.reward_game_won = 0
    env_cfg.reward_defeat_one_opponent = 0
    env_cfg.reward_detection = 0
    env_cfg.reward_idle = 0
    env_cfg.reward_collision_agent = 0
    env_cfg.reward_collision_obstacle = 0
    env_cfg.reward_damage = 1
    resolution = 84 #
    env_cfg.resolution_x = resolution
    env_cfg.resolution_y = resolution

    import jax.numpy as jnp
    env_cfg.cone_depth = 15.0
    env_cfg.cone_angle = jnp.pi * 1.99
    env_cfg.enable_waypoints = False
    env_cfg.use_stochastic_obs = False
    env_cfg.use_stochastic_comm = False
    env_cfg.max_agent_in_vec_obs = 100

    env_cfg.obs_type = "vector"
    env_cfg.discrete_actions = True


    device = torch.device("cuda:0")
    save_path = "gigastep/evaluation/trained_models/5v5_torch/"
    env_name = "identical_5_vs_5"

    paths = ["ppogigastepgiga20230531-101845.350990.pt",
             "ppogigastepgiga20230531-102104.363938.pt",
             "ppogigastepgiga20230531-102132.555251.pt",
             "ppogigastepgiga20230531-102159.562053.pt",
             "ppogigastepgiga20230531-102227.617178.pt",
             "ppogigastepgiga20230531-102324.301544.pt",
             "ppogigastepgiga20230531-102420.446652.pt",
             "ppogigastepgiga20230531-102643.081024.pt",
             "ppogigastepgiga20230531-115744.471220.pt",
            "ppogigastepgiga20230531-134318.816887.pt",
            "ppogigastepgiga20230531-142118.577104.pt",
            "ppogigastepgiga20230531-151346.319605.pt",
            "ppogigastepgiga20230531-165711.012685.pt"
            ]
    rate = Rating(device, save_path, env_name, env_cfg, num_match= 5)

    # for path in paths:
    #     actor_critic,_ = torch.load(
    #         "gigastep/evaluation/trained_models/5v5_torch/" + path,
    #         map_location=torch.device('cpu'))
    #
    #     rate_0 = rate.rate_policy(actor_critic,add_policy_to_pool= True)
    #     print(f"The uploaded policy got rating {rate_0}")
    #     ## after the pool has certain members

    for i in range(3):
        print(f"update policy pool {i+1} times")
        rate.update_pool_rating()


if __name__ == "__main__":
    main()


