import torch
import os
from enjoy_policy_discrete import evaluation_jax
import numpy as np
import bisect

class Rating():
    def __init__(self, device, save_path, env_name, env_cfg,
                 num_match=5):
        self.device = device
        self.save_path = save_path
        self.env_name = env_name
        self.env_cfg = env_cfg
        self.num_match = num_match
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if os.path.isfile(self.save_path + "rating_log.pt"):
            data = torch.load(self.save_path + "rating_log.pt")
            self.rating = data["rating"]
            self.path_pool = data["path_pool"]
        else:
            self.rating = []
            self.path_pool = []



    def update_pool_rating(self):
        pool_size = len(self.rating)

        for i in range(pool_size):
            print(f"Rating {i} in length of {pool_size}")
            for j in range(pool_size):
                if i!=j:
                    policy_0,_ = torch.load(self.path_pool[i])
                    policy_1,_ = torch.load(self.path_pool[j])
                    self.rating[i], self.rating[j], _ = self._match(
                        policy_0,
                        policy_1,
                        self.rating[i],
                        self.rating[j]
                    )
        print(f"rating list is{self.rating}")
        self._save_pool()

        
    
    def rate_policy(self, policy):

        policy.to(self.device)
        rating_0 = 1200
        if len(self.rating)==0:
            policy_1 = None
            rating_0, _, winning_0 = self._match(policy, policy_1,
                                      rating_0, 0)
        else:
            for i in range(len(self.rating)):
                policy_1,_ = torch.load(self.path_pool[i])
                rating_0, _, _= self._match(policy, policy_1,
                                          rating_0, self.rating[i])

        rank = bisect.bisect_left(sorted(self.rating), rating_0) + 1

        if len(self.rating)==0:
            if winning_0 == 1:
                position = len(self.rating)
                path = f"pool_{position:04}.pt"
                torch.save([policy, None], f"{self.save_path}{path}")
                
                self.rating.append(rating_0)
                self.path_pool.append(self.save_path + path)
                self._save_pool()
            else: 
                print(f"rating of agent is {rating_0} it is worse than base")

        else:
            if rating_0 > sorted(self.rating)[-1]:
                position = len(self.rating)
                self.rating.append(rating_0)
                path = f"pool_{position:04}.pt"
                torch.save([policy, None], f"{self.save_path}{path}")
                self.path_pool.append(self.save_path+path)
                self._save_pool()
            else:
                print(f"rating of agent is {rating_0}" +
                    f"which is lower than highest{sorted(self.rating)[-1]}" +
                    f" ranked in {rank}")

        return rating_0

    def _save_pool(self):
        data = {
                "rating": self.rating,
                "path_pool": self.path_pool
                }
        torch.save(data, self.save_path + "rating_log.pt")

    def _match(self, policy_0, policy_1, rate_0, rate_1):
        videos, winning_0 = evaluation_jax(self.env_name, obs_type="rgb",
                                           discrete_actions=True,
                                           actor_critic=policy_0,
                                           actor_critic_opponent=policy_1,
                                           num_of_evaluation=self.num_match,
                                           device=self.device,
                                           headless=True,
                                           env_cfg=self.env_cfg,
                                           show_num_agents=0)

        winning_0 = 1 if winning_0 > 0.5 else 0

        Exp_0 = 1 / (1 + 10 ** (rate_1 - rate_0))
        Exp_1 = 1 / (1 + 10 ** (rate_0 - rate_1))

        K = 16
        rate_0 = rate_0 + K * (winning_0 - Exp_0)
        rate_1 = rate_1 + K * (1 - winning_0 - Exp_1)

        return rate_0, rate_1, winning_0



if __name__ == "__main__":


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
    save_path = "/logs/pool/"
    env_name = "identical_2_vs_2"
    
    rate = Rating(device, save_path, env_name, env_cfg, num_match= 3)
    actor_critic,_ = torch.load("trained_models/ppogigastepgiga20230521-144137.667049.pt",
                            map_location=torch.device('cpu'))

    rate_0 = rate.rate_policy(actor_critic)

    ## after the pool has certain members
    rate.update_pool_rating()


    
    


