"""
Call with the following command
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python examples/train_gigastep.py
# Example usage: CUDA_VISIBLE_DEVICES=0  XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python examples/train_gigastep.py
"""
import argparse
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Box
from gigastep import make_scenario
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
import time
import jax
from gigastep import ScenarioBuilder

import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu" 
# os.environ["export XLA_PYTHON_CLIENT_PREALLOCATE"]= "false"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


# provide detailed information of each scenario
# mostly for policy sharing
policy_mapping_dict = {
    "identical_5_vs_5": {
        "description": "test",
        "team_prefix": ("red_", "blue_"),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
    "custom_1_vs_1": {
        "description": "test",
        "team_prefix": ("red_", "blue_"),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}

USE_VECTOR_OBS = True # False

# must inherited from MultiAgentEnv class
class RLlibMAGym(MultiAgentEnv):

    def __init__(self, env_config):
        print("================================instantiate================================")
        map = env_config.pop("map_name", None)

        if "custom" in map:
            assert map == "custom_1_vs_1"
            builder = ScenarioBuilder()
            builder.add_type(0, "default")
            builder.add_type(1, "default")
            if USE_VECTOR_OBS:
                self.env = builder.make(obs_type="vector")
            else:
                self.env = builder.make()
        else:
            if USE_VECTOR_OBS:
                self.env = make_scenario(map, obs_type="vector", **env_config)
            else:
                self.env = make_scenario(map, **env_config)
        self.rng = jax.random.PRNGKey(3)

        # assume all agent same action/obs space
        self.action_space = Box(
            low=np.array(self.env.action_space.low),
            high=np.array(self.env.action_space.high),
            shape=self.env.action_space.shape,
            dtype=self.env.action_space.dtype.type,
        )
        self.observation_space = GymDict({"obs": Box(
            low=np.array(self.env.observation_space.low),
            high=np.array(self.env.observation_space.high),
            shape=self.env.observation_space.shape,
            dtype=self.env.observation_space.dtype.type)})
        def _gen_agent_id(_prefix):
            return [f"{_prefix}{_i:02d}" for _i in range(self.env.n_agents//2)]
        self.agents = _gen_agent_id("red_") + _gen_agent_id("blue_")
        self.num_agents = self.env.n_agents

        env_config["map_name"] = map
        self.env_config = env_config

        self.max_steps = 500

    def reset(self):
        self.steps = 0

        self.rng, key_reset = jax.random.split(self.rng, 2)
        self.state, original_obs = self.env.reset(key_reset)

        obs = {}
        for i, name in enumerate(self.agents):
            obs[name] = {"obs": np.array(original_obs[i])}
        return obs

    def step(self, action_dict):
        assert list(action_dict.keys()) == self.agents # TODO: what if an agent die
        action_ls = [action_dict[key] for key in action_dict.keys()]
        action_jnp = jax.numpy.asarray(action_ls)

        self.rng, key_action, key_step = jax.random.split(self.rng, 3)
        self.state, o, r, d, ep_done = self.env.step(self.state, action_jnp, key_step)

        rewards = {}
        obs = {}
        for i, key in enumerate(action_dict.keys()):
            rewards[key] = r[i]
            obs[key] = {
                "obs": np.array(o[i])
            }

        # assert ep_done.item() == (sum(d).item() == self.num_agents)
        dones = {"__all__": ep_done.item()} # and (self.steps < self.max_steps)}

        self.steps += 1
        
        return obs, rewards, dones, {}

    def close(self):
        pass

    def render(self, mode=None):
        self.env.render()
        time.sleep(0.05)
        return True

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 100,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="mappo")
    parser.add_argument("--n-workers", type=int, default=1, choices=[0, 1])
    parser.add_argument("--n-envs", type=int, default=5)
    parser.add_argument("--local-mode", action="store_true", help="Set this if using GPU in JAX")
    parser.add_argument("--map-name", type=str, default="custom_1_vs_1",
                        choices=["custom_1_vs_1", "identical_5_vs_5"])
    args = parser.parse_args()

    # if not local_mode:
        # jax.distributed.initialize(num_processes=n_workers)
        # jax.distributed.initialize()
    # register new env
    ENV_REGISTRY["gigastep"] = RLlibMAGym
    COOP_ENV_REGISTRY["gigastep"] = RLlibMAGym
    # initialize env
    env = marl.make_env(environment_name="gigastep", map_name=args.map_name)
    # pick mappo algorithms
    if args.alg == "mappo":
        alg = marl.algos.mappo(hyperparam_source="test")
    elif args.alg == "ippo":
        alg = marl.algos.ippo(hyperparam_source="test")
    elif args.alg == "matrpo":
        alg = marl.algos.matrpo(hyperparam_source="test")
    elif args.alg == "maa2c":
        alg = marl.algos.maa2c(hyperparam_source="test")
    elif args.alg == "itrpo":
        alg = marl.algos.itrpo(hyperparam_source="test")
    elif args.alg == "ia2c":
        alg = marl.algos.ia2c(hyperparam_source="test")
    elif args.alg == "happo":
        alg = marl.algos.happo(hyperparam_source="test")
    elif args.alg == "hatrpo":
        alg = marl.algos.hatrpo(hyperparam_source="test")
    elif args.alg == "vdppo":
        alg = marl.algos.vdppo(hyperparam_source="test")
    elif args.alg == "vda2c":
        alg = marl.algos.vda2c(hyperparam_source="test")
    else:
        raise NotImplementedError
    # customize model
    model = marl.build_model(env, alg, {"core_arch": "gru", "hidden_state_size": 128})
    # start learning
    alg.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 1000000}, local_mode=args.local_mode, num_gpus=1,
            num_workers=args.n_workers, num_envs_per_worker=args.n_envs, share_policy='all', checkpoint_freq=50)
